#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  SPATIAL GI - Merge Pass (compute, cached, paired, lazy-payload)
//
//  All reconnections and visibility rays happen in the preceding shift pass;
//  this pass is pure data work and runs as a compute shader.
//
//  Per-slot MIS weights use only (partner.M, partner.W, GetPHat(partner.F))
//  loaded field-by-field from the reservoir. The full partner payload
//  (x2, n2_s, L2, V2, uv, matID, objID, eta, F) is only loaded when a
//  partner actually wins RIS — ~25-35% of slots in practice. This cuts the
//  dominant BW cost of the previous raygen merge.
//─────────────────────────────────────────────────────────────────────────────

// Scratch layout (8 + SPAT_COUNT_MAX*20 bytes/pixel; M_sum recomputed here):
//   [0]      uint  validCount
//   [4]      float my_Jc
//   [8 + s*20 + 0]   uint   nID
//   [8 + s*20 + 4]   float3 F         (MY shift contribution; visibility baked)
//   [8 + s*20 + 16]  float  Jn        (MY reconnection Jacobian)
static const uint SEL_STRIDE      = 8u + SPAT_COUNT_MAX * 20u;
static const uint SEL_SLOT_BASE   = 8u;
static const uint SEL_SLOT_STRIDE = 20u;

uint sel_addr(uint idx) { return idx * SEL_STRIDE; }
uint sel_slot_addr(uint idx, uint slot)
{
    return sel_addr(idx) + SEL_SLOT_BASE + slot * SEL_SLOT_STRIDE;
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

    Reservoir rdi = loadReservoir(g_Reservoirs_current, pixelIdx);

    // Emitter early-out: no reuse
    if (load_isEmitter(g_sample_current, pixelIdx))
    {
        storeReservoir(g_Reservoirs_last, pixelIdx, rdi);
        return;
    }

    // Disabled early-out: canonical passthrough
    if (!(rs_flags & 8u))
    {
        const float  W = (rdi.W > 0.0f) ? rdi.W : 0.0f;
        gScratchPing[uint3(launchIndex, 2)] = float4(rdi.F * W, 0);
        storeReservoir(g_Reservoirs_last, pixelIdx, rdi);
        return;
    }

    //─────────────────────────────────────────────────────────────────────────
    // Load scratch header (8 bytes)
    //─────────────────────────────────────────────────────────────────────────
    const uint  baseAddr = sel_addr(pixelIdx);
    const uint2 header   = g_pathStateBuffer.Load2(baseAddr);
    const uint  validCount = header.x;
    const float my_Jc      = asfloat(header.y);

    //─────────────────────────────────────────────────────────────────────────
    // Canonical contribution (no reconnection — uses stored F)
    //─────────────────────────────────────────────────────────────────────────
    const float M_c = min(SPAT_MCAP, rdi.M);
    rdi.M = M_c;

    const float  visReuse_c    = (rdi.W > 0.0f) ? 1.0f : 0.0f;
    const float  p_c           = GetPHat(rdi.F) * visReuse_c;
    float3       contrib_final = rdi.F * visReuse_c;

    if (validCount == 0u)
    {
        gScratchPing[uint3(launchIndex, 2)] = float4(contrib_final * rdi.W, 0);
        storeReservoir(g_Reservoirs_last, pixelIdx, rdi);
        return;
    }

    //─────────────────────────────────────────────────────────────────────────
    // Gather pass: read per-slot partner info cheaply — nID, partner's Jc,
    // partner's shift-to-me (F_mag + Jn), partner's M. Accumulate M_sum.
    //─────────────────────────────────────────────────────────────────────────
    uint  slot_nID       [SPAT_COUNT_MAX];
    float slot_partner_Jc[SPAT_COUNT_MAX];
    float slot_partner_Mn[SPAT_COUNT_MAX];
    float slot_p_hat_ptm [SPAT_COUNT_MAX];  // partner -> me density at my x2

    float M_sum = M_c;

    [unroll]
    for (uint i = 0u; i < SPAT_COUNT_MAX; ++i)
    {
        slot_nID       [i] = 0xFFFFFFFFu;
        slot_partner_Jc[i] = 0.0f;
        slot_partner_Mn[i] = 0.0f;
        slot_p_hat_ptm [i] = 0.0f;

        const uint nID = g_pathStateBuffer.Load(sel_slot_addr(pixelIdx, i));
        if (nID == 0xFFFFFFFFu) continue;

        // Partner's cached shift-to-me at their slot i: float3 F + float Jn
        const uint4  pShift  = g_pathStateBuffer.Load4(sel_slot_addr(nID, i) + 4u);
        const float3 p_F     = asfloat(pShift.xyz);
        const float  p_F_mag = GetPHat(p_F);
        const float  p_Jn    = asfloat(pShift.w);

        // Partner's Jc from their scratch header
        const float p_Jc = asfloat(g_pathStateBuffer.Load(sel_addr(nID) + 4u));

        // Partner's M from their reservoir (1-field load)
        const float Mn = min(SPAT_MCAP, load_M(g_Reservoirs_current, nID));

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
    for (uint j = 0u; j < SPAT_COUNT_MAX; ++j)
    {
        const float Mn = slot_partner_Mn[j];
        if (Mn <= 0.0f) continue;
        const float m_den = m_num_c + (M_sum - M_c) * slot_p_hat_ptm[j];
        if (m_den > EPSILON)
        {
            mis_c += (Mn / max(M_sum, 1.0f)) * (m_num_c / m_den);
        }
    }

    rdi.w_sum = mis_c * p_c * rdi.W;

    //─────────────────────────────────────────────────────────────────────────
    // RIS — inline update; lazy-load partner payload only on acceptance.
    //─────────────────────────────────────────────────────────────────────────
    uint2 seed = GetSeed(pixelIdx, time, 2);

    [loop]
    for (uint k = 0u; k < SPAT_COUNT_MAX; ++k)
    {
        const uint nID = slot_nID[k];
        if (nID == 0xFFFFFFFFu) continue;

        // MY shift for slot k: float3 F (offset +4) + float Jn (offset +16)
        const uint4  myShift = g_pathStateBuffer.Load4(sel_slot_addr(pixelIdx, k) + 4u);
        const float3 my_F    = asfloat(myShift.xyz);
        const float  my_F_mag= GetPHat(my_F);
        const float  my_Jn   = asfloat(myShift.w);

        const float p_hat_me_to_partner =
            my_F_mag * JacobianRatio(my_Jn, slot_partner_Jc[k]);

        // Two fields from partner's reservoir for MIS (W, F magnitude)
        const float partner_W     = load_W(g_Reservoirs_current, nID);
        const float partner_F_mag = GetPHat(load_F(g_Reservoirs_current, nID));
        const float Mn            = slot_partner_Mn[k];

        const float mis_n = PairwiseMIS_Neighbor_Spat(
            M_sum, M_c, Mn,
            p_hat_me_to_partner,
            partner_W, partner_F_mag);

        const float w_n = mis_n * p_hat_me_to_partner * partner_W;

        // Inline RIS step — matches UpdateReservoir byte-for-byte
        rdi.w_sum += w_n;
        rdi.M     += (uint)Mn;

        if (RandomFloatSingle(seed.x) < (w_n / rdi.w_sum))
        {
            // Lazy payload load: only pay the cost on RIS acceptance.
            // pack1 is fetched once for x2 + n2_s; other fields are 4 B each.
            const uint   p_objID = load_objID(g_Reservoirs_current, nID);
            const uint4  pack1   = g_Reservoirs_current.Load4(addr_pack1(nID));

            rdi.x2    = ObjectToWorldPos(p_objID, asfloat(pack1.xyz));
            rdi.n2_s  = ObjectToWorldNrm(p_objID, UnpackNormal(pack1.w));
            rdi.objID = p_objID;
            rdi.matID = load_matID(g_Reservoirs_current, nID);
            rdi.eta   = load_eta  (g_Reservoirs_current, nID);

            rdi.L2    = load_L2(g_Reservoirs_current, nID);
            rdi.V2    = load_V2(g_Reservoirs_current, nID);
            rdi.uv    = load_uv_res(g_Reservoirs_current, nID);
            // rdi.F is overwritten below with the shift-to-me contribution
            // (contrib_final), which IS the correct target for this pixel.

            contrib_final = my_F;
        }
    }

    //─────────────────────────────────────────────────────────────────────────
    // Finalize: store full float3 contribution in rdi.F; GetPHat(F) IS the
    // target magnitude (invariant holds exactly — no RGB9E5 round-trip).
    //─────────────────────────────────────────────────────────────────────────
    rdi.F = contrib_final;
    const float F_mag_final = GetPHat(rdi.F);

    if (F_mag_final > EPSILON && rdi.w_sum > 0.0f && rdi.w_sum < 1e10f)
    {
        float W = rdi.w_sum / F_mag_final;
        if (isnan(W) || isinf(W) || (W < 0.0f)) W = 0.0f;
        rdi.W = W;
    }
    else
    {
        rdi.W = 0.0f;
    }

    gScratchPing[uint3(launchIndex, 2)] = float4(rdi.F * rdi.W, 0);
    storeReservoir(g_Reservoirs_last, pixelIdx, rdi);
}
