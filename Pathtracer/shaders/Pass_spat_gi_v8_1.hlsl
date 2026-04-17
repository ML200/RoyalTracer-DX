#include "Includes_v8.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  SPATIAL GI - Merge Pass (cached, paired)
//
//  All reconnections and visibility rays happen in the preceding shift pass.
//  This pass only reads cached shift payloads (my-side for RIS, partner-side
//  for MIS denominator) and does the MIS + RIS arithmetic.
//─────────────────────────────────────────────────────────────────────────────

static const uint GI_SEL_STRIDE      = 64u;
static const uint GI_SEL_SLOT_BASE   = 16u;
static const uint GI_SEL_SLOT_STRIDE = 16u;

uint gi_sel_addr(uint idx) { return idx * GI_SEL_STRIDE; }
uint gi_sel_slot_addr(uint idx, uint slot)
{
    return gi_sel_addr(idx) + GI_SEL_SLOT_BASE + slot * GI_SEL_SLOT_STRIDE;
}

[shader("raygeneration")]
void Pass_spat_gi_v8_1()
{
    uint sortKey;
    {
        const uint2 li  = DispatchRaysIndex().xy;
        const uint  px  = MapPixelID(float2(IMG_W, IMG_H), li);
        const bool  emi = load_isEmitter(g_sample_current, px);

        if (emi || !(rs_flags & 8u))
        {
            sortKey = emi ? 0u : 1u;
        }
        else
        {
            sortKey = 2u + g_pathStateBuffer.Load(gi_sel_addr(px)); // validCount
        }
    }
    dx::MaybeReorderThread(sortKey, 3);

    //════════════════════════════════════════════════════════════════════════
    const uint2  launchIndex = DispatchRaysIndex().xy;
    const float2 dims        = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims, launchIndex);

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
    // Load scratch header
    //─────────────────────────────────────────────────────────────────────────
    const uint baseAddr = gi_sel_addr(pixelIdx);
    const uint3 header  = g_pathStateBuffer.Load3(baseAddr);
    const uint  validCount = header.x;
    const float M_sum_nbr  = asfloat(header.y);
    const float my_Jc      = asfloat(header.z);

    //─────────────────────────────────────────────────────────────────────────
    // Canonical contribution (p_c)
    //─────────────────────────────────────────────────────────────────────────
    const float M_c = min(SPAT_MCAP_GI, rdi.M_gi);
    rdi.M_gi = M_c;
    const float M_sum = M_sum_nbr + M_c;

    const float  visReuse_c   = (rdi.W_gi > 0.0f) ? 1.0f : 0.0f;
    const float  p_c          = rdi.F_mag_gi * visReuse_c;
    float3       contrib_final = UnpackRGB9E5(rdi.F_gi) * rdi.F_mag_gi * visReuse_c;

    //─────────────────────────────────────────────────────────────────────────
    // No partners -> canonical passthrough
    //─────────────────────────────────────────────────────────────────────────
    if (validCount == 0u)
    {
        gScratchPing[uint3(launchIndex, 2)] = float4(contrib_final * rdi.W_gi, 0);
        storeReservoirGI(g_Reservoirs_last_gi, pixelIdx, rdi);
        return;
    }

    //─────────────────────────────────────────────────────────────────────────
    // Loop 1: canonical MIS accumulation from partners cached shifts.
    //─────────────────────────────────────────────────────────────────────────
    float       mis_c   = M_c / max(M_sum, 1.0f);
    const float m_num_c = M_c * p_c;

    [loop]
    for (uint i = 0u; i < SPAT_COUNT_MAX_GI; ++i)
    {
        const uint mySlotAddr = gi_sel_slot_addr(pixelIdx,i);
        const uint nID = g_pathStateBuffer.Load(mySlotAddr);
        if (nID == 0xFFFFFFFFu) continue;

        // Partners cached shift-to-me at their slot i
        const uint  pSlotAddr = gi_sel_slot_addr(nID, i);
        const uint2 pShift    = g_pathStateBuffer.Load2(pSlotAddr + 8u);
        const float p_F_mag   = asfloat(pShift.x);
        const float p_Jn      = asfloat(pShift.y);

        const float p_hat_from_i = p_F_mag * JacobianRatio(p_Jn, my_Jc);
        const float m_den        = m_num_c + (M_sum - M_c) * p_hat_from_i;

        if (m_den > EPSILON)
        {
            const float M_i = min(SPAT_MCAP_GI,
                                  load_M_gi(g_Reservoirs_current_gi, nID));
            mis_c += (M_i / max(M_sum, 1.0f)) * (m_num_c / m_den);
        }
    }

    rdi.w_sum_gi = mis_c * p_c * rdi.W_gi;

    //─────────────────────────────────────────────────────────────────────────
    // Loop 2: RIS update per partner
    //─────────────────────────────────────────────────────────────────────────
    uint2 seed = GetSeed(pixelIdx, time, 2);

    [loop]
    for (uint k = 0u; k < SPAT_COUNT_MAX_GI; ++k)
    {
        const uint  mySlotAddr = gi_sel_slot_addr(pixelIdx,k);
        const uint4 myShift    = g_pathStateBuffer.Load4(mySlotAddr);
        const uint  nID        = myShift.x;
        if (nID == 0xFFFFFFFFu) continue;

        const uint  my_F_pack = myShift.y;
        const float my_F_mag  = asfloat(myShift.z);
        const float my_Jn     = asfloat(myShift.w);

        const float partner_Jc = asfloat(
            g_pathStateBuffer.Load(gi_sel_addr(nID) + 8u));

        const float p_hat_me_to_partner =
            my_F_mag * JacobianRatio(my_Jn, partner_Jc);

        const Reservoir_GI rdi_r = loadReservoirGI(g_Reservoirs_current_gi, nID);
        const float Mn = min(SPAT_MCAP_GI, rdi_r.M_gi);

        const float mis_n = PairwiseMIS_Neighbor_Spat_GI(
            M_sum, M_c, Mn,
            p_hat_me_to_partner,
            rdi_r.W_gi, rdi_r.F_mag_gi);

        const float w_n = mis_n * p_hat_me_to_partner * rdi_r.W_gi;

        const float3 contrib_n = UnpackRGB9E5(my_F_pack) * my_F_mag;

        if (UpdateReservoirGI(
                rdi, w_n, Mn,
                rdi_r.x2_gi, rdi_r.n2_s_gi, rdi_r.L2_gi, rdi_r.V2_gi,
                rdi_r.uv_gi,
                rdi_r.matID_gi, rdi_r.objID_gi, rdi_r.eta_gi,
                rdi_r.F_gi, rdi_r.F_mag_gi,
                seed))
        {
            contrib_final = contrib_n;
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
