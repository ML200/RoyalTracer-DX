#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//SPATIAL GI MERGE PASS
//====================================
//pure data, all rays already cast in shift, payload lazy loaded on RIS acceptance

//scratch layout mirrors Pass_spat_gi_select_v8.hlsl
//slot is [nID(4) | F(12) | Jn(4) | Jc(4)], Jc per-slot for sector locality
static const uint SEL_STRIDE      = 4u + SPAT_COUNT_MAX * 24u;
static const uint SEL_SLOT_BASE   = 4u;
static const uint SEL_SLOT_STRIDE = 24u;

uint sel_addr(uint idx) { return idx * SEL_STRIDE; }
uint sel_slot_addr(uint idx, uint slot)
{
    return sel_addr(idx) + SEL_SLOT_BASE + slot * SEL_SLOT_STRIDE;
}

//====================================
//MERGE PASS ENTRY
//====================================
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    const uint2  launchIndex = tid.xy;
    const float2 dims        = float2(IMG_W, IMG_H);
    const uint   pixelIdx    = MapPixelID(dims, launchIndex);

    //cell-mode active: Pass_spat_gi_cell_v8 owns 100% of spatial reuse (incl.
    //emitter / passthrough cases), so this pass does nothing and writes nothing.
    if (CELL_SPATIAL_MODE) return;

    //G-buffer copy for temporal reuse removed: the renderer ping-pongs the
    //current/last sample-buffer bindings per frame (Renderer::SwapSampleBuffers)

    //emitter early out, no reuse. Next frame's temporal rejects emitter
    //pixels via the sample-buffer flag before touching their reservoir, so
    //the full canonical round trip the old path did is dead weight - only
    //the V2 word needs to stay inert because the dup pass streams it
    //unconditionally from the history buffer.
    if (load_isEmitter(g_sample_current, pixelIdx))
    {
        g_Reservoirs_last.Store(addr_v2(pixelIdx), PROBE_DI_NORMAL_ZERO_CODE);
        return;
    }

    Reservoir rdi = loadReservoir(g_Reservoirs_current, pixelIdx);

    //disabled early terrain, canonical passthrough
    if (!(rs_flags & 8u))
    {
        const float  W = (rdi.W > 0.0f) ? rdi.W : 0.0f;
        gScratchPing[uint3(launchIndex, 2)] = float4(rdi.F * W, 0);
        storeReservoir(g_Reservoirs_last, pixelIdx, rdi);
        return;
    }

    //====================================
    //LOAD SCRATCH HEADER
    //====================================
    //header is just validCount now, my_Jc moved into slot 0 (see shift pass)
    const uint  baseAddr   = sel_addr(pixelIdx);
    const uint  validCount = g_pathStateBuffer.Load(baseAddr);

    //====================================
    //CANONICAL CONTRIBUTION
    //====================================
    //no reconnection, uses stored F
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

    //====================================
    //GATHER PASS
    //====================================
    //cheap per slot, nID, partner Jc, partner shift to me, partner M
    uint  slot_nID       [SPAT_COUNT_MAX];
    float slot_partner_Jc[SPAT_COUNT_MAX];
    float slot_partner_Mn[SPAT_COUNT_MAX];
    float slot_p_hat_ptm [SPAT_COUNT_MAX];

    float M_sum = M_c;

    //my Jc lives in slot 0's Jc field, written by the shift pass. Tile coalesced
    //SELF load before the scattered partner loads in the unrolled body below.
    const float my_Jc = asfloat(g_pathStateBuffer.Load(sel_slot_addr(pixelIdx, 0u) + 20u));

    [unroll]
    for (uint i = 0u; i < SPAT_COUNT_MAX; ++i)
    {
        slot_nID       [i] = 0xFFFFFFFFu;
        slot_partner_Jc[i] = 0.0f;
        slot_partner_Mn[i] = 0.0f;
        slot_p_hat_ptm [i] = 0.0f;

        const uint nID = g_pathStateBuffer.Load(sel_slot_addr(pixelIdx, i));
        if (nID == 0xFFFFFFFFu) continue;

        //partner slot record is [nID | F | Jn | Jc]; head fetch covers
        //nID + F, tail fetch Jn + Jc - same sector locality as before.
        //MUTUALITY CHECK on the partner's nID: the reuse textures are
        //self-inverting so partner slot i should name me back, but select
        //rejection can be asymmetric, in which case the partner never wrote
        //F/Jn/Jc for this slot this frame and the words here are stale. A
        //partner my select accepted always ran its own slot loop (it is
        //non-emitter with M>0), so its nID word is fresh - if it is not me,
        //drop the slot from M_sum and the RIS loop.
        const uint   partnerSlot = sel_slot_addr(nID, i);
        const uint4  pHead   = g_pathStateBuffer.Load4(partnerSlot);
        if (pHead.x != pixelIdx) continue;
        const uint2  pTail   = g_pathStateBuffer.Load2(partnerSlot + 16u);
        const float3 p_F     = asfloat(pHead.yzw);
        const float  p_F_mag = GetPHat(p_F);
        const float  p_Jn    = asfloat(pTail.x);
        const float  p_Jc    = asfloat(pTail.y);

        //one field partner M
        const float Mn = min(SPAT_MCAP, load_M(g_Reservoirs_current, nID));

        slot_nID       [i] = nID;
        slot_partner_Jc[i] = p_Jc;
        slot_partner_Mn[i] = Mn;
        slot_p_hat_ptm [i] = p_F_mag * JacobianRatio(p_Jn, my_Jc);

        M_sum += Mn;
    }

    //====================================
    //CANONICAL MIS
    //====================================
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

    //====================================
    //RIS INLINE UPDATE
    //====================================
    //partner payload lazy loaded only on acceptance
    uint2 seed = GetSeed(pixelIdx, time, 2);

    [loop]
    for (uint k = 0u; k < SPAT_COUNT_MAX; ++k)
    {
        const uint nID = slot_nID[k];
        if (nID == 0xFFFFFFFFu) continue;

        //my shift for slot k
        const uint4  myShift = g_pathStateBuffer.Load4(sel_slot_addr(pixelIdx, k) + 4u);
        const float3 my_F    = asfloat(myShift.xyz);
        const float  my_F_mag= GetPHat(my_F);
        const float  my_Jn   = asfloat(myShift.w);

        const float p_hat_me_to_partner =
            my_F_mag * JacobianRatio(my_Jn, slot_partner_Jc[k]);

        //partner F + W ride one 16B group - single fetch for the MIS pair
        const uint4 pFW           = g_Reservoirs_current.Load4(addr_fw(nID));
        const float partner_W     = asfloat(pFW.w);
        const float partner_F_mag = GetPHat(asfloat(pFW.xyz));
        const float Mn            = slot_partner_Mn[k];

        const float mis_n = PairwiseMIS_Neighbor_Spat(
            M_sum, M_c, Mn,
            p_hat_me_to_partner,
            partner_W, partner_F_mag);

        const float w_n = mis_n * p_hat_me_to_partner * partner_W;

        //inline RIS step, matches UpdateReservoir byte for byte
        rdi.w_sum += w_n;
        rdi.M     += (uint)Mn;

        if (RandomFloatSingle(seed.x) < (w_n / rdi.w_sum))
        {
            //lazy payload load only on RIS acceptance - 4 grouped fetches,
            //leaves the accumulated F/W/M/w_sum untouched.
            //rdi.F overwritten below with shift to me contrib
            loadReservoirPayload(g_Reservoirs_current, nID, rdi);

            contrib_final = my_F;
        }
    }

    //====================================
    //FINALIZE
    //====================================
    //full RGB contribution in rdi.F, target magnitude is GetPHat(F)
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
