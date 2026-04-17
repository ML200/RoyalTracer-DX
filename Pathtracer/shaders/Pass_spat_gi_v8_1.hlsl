#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  SPATIAL GI - Merge Pass (compute, cached, paired, lazy-payload)
//
//  All reconnections and visibility rays happen in the preceding shift pass;
//  this pass is pure data work and runs as a compute shader.
//
//  Per-slot MIS weights use only (partner.M, partner.W, partner.F_mag_gi)
//  loaded field-by-field from the reservoir. The full partner payload
//  (x2, n2_s, L2, V2, uv, matID, objID, eta, F_gi) is only loaded when a
//  partner actually wins RIS — ~25-35% of slots in practice. This cuts the
//  dominant BW cost of the previous raygen merge.
//─────────────────────────────────────────────────────────────────────────────

// Scratch layout (56 bytes/pixel; M_sum removed vs. prior version):
//   [0]      uint  validCount
//   [4]      float my_Jc
//   [8 + s*16 + 0]  uint  nID
//   [8 + s*16 + 4]  uint  F_pack      (MY shift color, RGB9E5)
//   [8 + s*16 + 8]  float F_mag       (MY shift magnitude, visibility baked)
//   [8 + s*16 +12]  float Jn          (MY reconnection Jacobian)
static const uint GI_SEL_STRIDE      = 56u;
static const uint GI_SEL_SLOT_BASE   = 8u;
static const uint GI_SEL_SLOT_STRIDE = 16u;

uint gi_sel_addr(uint idx) { return idx * GI_SEL_STRIDE; }
uint gi_sel_slot_addr(uint idx, uint slot)
{
    return gi_sel_addr(idx) + GI_SEL_SLOT_BASE + slot * GI_SEL_SLOT_STRIDE;
}

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    const uint2  launchIndex = tid.xy;
    const float2 dims        = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims, launchIndex);

    // Copy compact G-buffer for temporal reuse
    copySampleData(g_sample_last, g_sample_current, pixelIdx);

    Reservoir_GI rdi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);

    // Emitter early-out: no reuse
    if (load_isEmitter(g_sample_current, pixelIdx))
    {
        storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
        return;
    }

    // Disabled early-out: canonical passthrough
    if (!(rs_flags & 8u))
    {
        const float3 c = UnpackRGB9E5(rdi.F_gi) * rdi.F_mag_gi;
        const float  W = (rdi.W_gi > 0.0f) ? rdi.W_gi : 0.0f;
        gScratchPing[uint3(launchIndex, 2)] = float4(c * W, 0);
        storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
        return;
    }

    //─────────────────────────────────────────────────────────────────────────
    // Load scratch header (8 bytes)
    //─────────────────────────────────────────────────────────────────────────
    const uint  baseAddr = gi_sel_addr(pixelIdx);
    const uint2 header   = g_pathStateBuffer.Load2(baseAddr);
    const uint  validCount = header.x;
    const float my_Jc      = asfloat(header.y);

    //─────────────────────────────────────────────────────────────────────────
    // Canonical contribution (no reconnection — uses stored F)
    //─────────────────────────────────────────────────────────────────────────
    const float M_c = min(SPAT_MCAP_GI, rdi.M_gi);
    rdi.M_gi = M_c;

    const float  visReuse_c    = (rdi.W_gi > 0.0f) ? 1.0f : 0.0f;
    const float  p_c           = rdi.F_mag_gi * visReuse_c;
    float3       contrib_final = UnpackRGB9E5(rdi.F_gi) * rdi.F_mag_gi * visReuse_c;

    if (validCount == 0u)
    {
        gScratchPing[uint3(launchIndex, 2)] = float4(contrib_final * rdi.W_gi, 0);
        storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
        return;
    }

    //─────────────────────────────────────────────────────────────────────────
    // Gather pass: read per-slot partner info cheaply — nID, partner's Jc,
    // partner's shift-to-me (F_mag + Jn), partner's M. Accumulate M_sum.
    //─────────────────────────────────────────────────────────────────────────
    uint  slot_nID       [SPAT_COUNT_MAX_GI];
    float slot_partner_Jc[SPAT_COUNT_MAX_GI];
    float slot_partner_Mn[SPAT_COUNT_MAX_GI];
    float slot_p_hat_ptm [SPAT_COUNT_MAX_GI];  // partner -> me density at my x2

    float M_sum = M_c;

    [unroll]
    for (uint i = 0u; i < SPAT_COUNT_MAX_GI; ++i)
    {
        slot_nID       [i] = 0xFFFFFFFFu;
        slot_partner_Jc[i] = 0.0f;
        slot_partner_Mn[i] = 0.0f;
        slot_p_hat_ptm [i] = 0.0f;

        const uint nID = g_pathStateBuffer.Load(gi_sel_slot_addr(pixelIdx, i));
        if (nID == 0xFFFFFFFFu) continue;

        // Partner's cached shift-to-me at their slot i (F_mag + Jn)
        const uint2 pShift = g_pathStateBuffer.Load2(gi_sel_slot_addr(nID, i) + 8u);
        const float p_F_mag = asfloat(pShift.x);
        const float p_Jn    = asfloat(pShift.y);

        // Partner's Jc from their scratch header
        const float p_Jc = asfloat(g_pathStateBuffer.Load(gi_sel_addr(nID) + 4u));

        // Partner's M from their reservoir (1-field load)
        const float Mn = min(SPAT_MCAP_GI, load_M_gi(g_Reservoirs_current_gi, nID));

        slot_nID       [i] = nID;
        slot_partner_Jc[i] = p_Jc;
        slot_partner_Mn[i] = Mn;
        slot_p_hat_ptm [i] = p_F_mag * JacobianRatio(p_Jn, my_Jc);

        M_sum += Mn;
    }

    //─────────────────────────────────────────────────────────────────────────
    // Canonical MIS (cached, O(N))
    //─────────────────────────────────────────────────────────────────────────
    float       mis_c   = M_c / max(M_sum, 1.0f);
    const float m_num_c = M_c * p_c;

    [unroll]
    for (uint j = 0u; j < SPAT_COUNT_MAX_GI; ++j)
    {
        const float Mn = slot_partner_Mn[j];
        if (Mn <= 0.0f) continue;
        const float m_den = m_num_c + (M_sum - M_c) * slot_p_hat_ptm[j];
        if (m_den > EPSILON)
        {
            mis_c += (Mn / max(M_sum, 1.0f)) * (m_num_c / m_den);
        }
    }

    rdi.w_sum_gi = mis_c * p_c * rdi.W_gi;

    //─────────────────────────────────────────────────────────────────────────
    // RIS — inline update; lazy-load partner payload only on acceptance.
    //─────────────────────────────────────────────────────────────────────────
    uint2 seed = GetSeed(pixelIdx, time, 2);

    [loop]
    for (uint k = 0u; k < SPAT_COUNT_MAX_GI; ++k)
    {
        const uint nID = slot_nID[k];
        if (nID == 0xFFFFFFFFu) continue;

        // MY shift for slot k
        const uint4 myShift   = g_pathStateBuffer.Load4(gi_sel_slot_addr(pixelIdx, k));
        const uint  my_F_pack = myShift.y;
        const float my_F_mag  = asfloat(myShift.z);
        const float my_Jn     = asfloat(myShift.w);

        const float p_hat_me_to_partner =
            my_F_mag * JacobianRatio(my_Jn, slot_partner_Jc[k]);

        // Two fields from partner's reservoir for MIS (W, F_mag)
        const float partner_W     = load_W_gi    (g_Reservoirs_current_gi, nID);
        const float partner_F_mag = load_F_mag_gi(g_Reservoirs_current_gi, nID);
        const float Mn            = slot_partner_Mn[k];

        const float mis_n = PairwiseMIS_Neighbor_Spat_GI(
            M_sum, M_c, Mn,
            p_hat_me_to_partner,
            partner_W, partner_F_mag);

        const float w_n = mis_n * p_hat_me_to_partner * partner_W;

        // Inline RIS step — matches UpdateReservoirGI byte-for-byte
        rdi.w_sum_gi += w_n;
        rdi.M_gi     += (uint)Mn;

        if (RandomFloatSingle(seed.x) < (w_n / rdi.w_sum_gi))
        {
            // Lazy payload load: only pay the cost on RIS acceptance.
            // pack1 is fetched once for x2 + n2_s; other fields are 4 B each.
            const uint   p_objID = load_objID_gi(g_Reservoirs_current_gi, nID);
            const uint4  pack1   = g_Reservoirs_current_gi.Load4(gi_addr_pack1(nID));

            rdi.x2_gi    = ObjectToWorldPos(p_objID, asfloat(pack1.xyz));
            rdi.n2_s_gi  = ObjectToWorldNrm(p_objID, UnpackNormal(pack1.w));
            rdi.objID_gi = p_objID;
            rdi.matID_gi = load_matID_gi(g_Reservoirs_current_gi, nID);
            rdi.eta_gi   = load_eta_gi  (g_Reservoirs_current_gi, nID);

            rdi.L2_gi    = load_L2_gi(g_Reservoirs_current_gi, nID);
            rdi.V2_gi    = load_V2_gi(g_Reservoirs_current_gi, nID);
            rdi.uv_gi    = load_uv_gi(g_Reservoirs_current_gi, nID);
            rdi.F_gi     = load_F_gi (g_Reservoirs_current_gi, nID);
            rdi.F_mag_gi = partner_F_mag;  // already loaded above

            contrib_final = UnpackRGB9E5(my_F_pack) * my_F_mag;
        }
    }

    //─────────────────────────────────────────────────────────────────────────
    // Finalize: normalize W, pack F from contrib_final
    //─────────────────────────────────────────────────────────────────────────
    const float  F_mag_out  = GetPHat(contrib_final);
    const float3 F_norm_out = (F_mag_out > 1e-20f) ? contrib_final / F_mag_out : float3(0, 0, 0);

    if (F_mag_out > EPSILON && rdi.w_sum_gi > 0.0f && rdi.w_sum_gi < 1e10f)
    {
        float W = rdi.w_sum_gi / F_mag_out;
        if (isnan(W) || isinf(W) || (W < 0.0f)) W = 0.0f;
        rdi.W_gi = W;
    }
    else
    {
        rdi.W_gi = 0.0f;
    }

    rdi.F_gi     = PackRGB9E5(F_norm_out);
    rdi.F_mag_gi = F_mag_out;

    gScratchPing[uint3(launchIndex, 2)] = float4(contrib_final * rdi.W_gi, 0);
    storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
}
