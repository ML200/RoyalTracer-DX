#define COMPUTE_PASS
#define SPMIS_GRID_NONCOHERENT
#include "Includes_v8.hlsli"
#include "RestirLite_v8.hlsli"

//====================================
//PAIRED SPATIAL MERGE AND SHADING
//====================================
// Defensive pairwise MIS over the pixel's partners (both directions of each
// pair come from the shift pass), one resampling step and exact shading of
// the winner into the radiance estimate. Lite reservoirs last only this frame.
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;
    const uint2 pixel = tid.xy;
    const uint2 dims = uint2(IMG_W, IMG_H);
    const uint px = MapPixelID(dims, (int2)pixel);
    if (load_flagsWord(g_sample_current, px) & SD_FLAG_NOBOUNCE) return;
    LiteReservoir rc = LiteLoad(g_Reservoirs_current, px);
    if (rc.M == 0u) return;
    const SDRecord sd = load_SD(g_sample_current, px);
    const LiteReceiver rcv = LiteReceiverFromSD(sd);
    const float cap = (float)max(lite_spatMcap, 1u);
    const float Mc = min((float)rc.M, cap);

    float3 yc = 0.0f;
    float3 nyc = 0.0f;
    LiteLink lc = LiteLinkFrom(0.0f, 0.0f, 0.0f, false);
    float pcc = 0.0f;
    if (LiteHasSample(rc.s))
    {
        yc = LiteWorldPosition(rc.s);
        nyc = LiteWorldNormal(rc.s);
        lc = LiteConnect(rcv, rc.s, yc, nyc);
        pcc = LiteTarget(rcv.albedo, rc.s, lc, yc, LITE_UNSHADOWED ? 1.0f : Luma(rc.tint));
    }

    //====================================
    //GATHER PARTNERS
    //====================================
    // p_ci: my target at the partner's sample (my shift slot), p_ic: the
    // partner's target at my sample (its shift slot of the same index),
    // p_ii: the partner's target at its own sample.
    uint  qpx[LITE_SLOTS_MAX];
    float Mi[LITE_SLOTS_MAX], pci[LITE_SLOTS_MAX], pic[LITE_SLOTS_MAX], pii[LITE_SLOTS_MAX], Wi[LITE_SLOTS_MAX];
    uint  visPk[LITE_SLOTS_MAX];
    float Msum = Mc;
    const uint mask = LITE_SPATIAL ? g_pathStateBuffer.Load(LiteShiftMaskAddress(px)) : 0u;
    [unroll]
    for (uint s = 0u; s < LITE_SLOTS_MAX; ++s)
    {
        qpx[s] = 0xffffffffu;
        Mi[s] = 0.0f; pci[s] = 0.0f; pic[s] = 0.0f; pii[s] = 0.0f; Wi[s] = 0.0f;
        visPk[s] = 0u;
        if ((mask & (1u << s)) == 0u) continue;
        const uint3 mine = g_pathStateBuffer.Load3(LiteShiftAddress(px, s));
        const uint p = mine.z;
        const uint2 partner = g_Reservoirs_current.Load2(LiteAddress(p) + 24u); // W | meta
        qpx[s] = p;
        Mi[s] = min((float)(partner.y & 255u), cap);
        Wi[s] = asfloat(partner.x);
        pci[s] = asfloat(mine.x);
        visPk[s] = mine.y;
        pic[s] = asfloat(g_pathStateBuffer.Load(LiteShiftAddress(p, s)));
        pii[s] = asfloat(g_pathStateBuffer.Load(LiteShiftOwnAddress(p)));
        Msum += Mi[s];
    }

    //====================================
    //PAIRWISE MIS AND RESAMPLING
    //====================================
    const float O = Msum - Mc;
    float mc = Mc / Msum;
    [unroll]
    for (uint j = 0u; j < LITE_SLOTS_MAX; ++j)
        if (qpx[j] != 0xffffffffu) mc += LiteMisCanonicalTerm(Mi[j], pcc, Mc, pic[j], O, Msum);
    float wsum = mc * pcc * rc.W;
    int selected = -1;
    float phatSel = pcc;
    uint seed = GetSeed(pixel, sharc_frame, 0x53504154u).x;
    [unroll]
    for (uint k = 0u; k < LITE_SLOTS_MAX; ++k)
    {
        if (qpx[k] == 0xffffffffu) continue;
        const float mi = LiteMisPartner(Mi[k], pii[k], Mc, pci[k], O, Msum);
        const float wi = mi * pci[k] * Wi[k];
        if (!(wi > 0.0f)) continue;
        wsum += wi;
        if (RandomFloatPCG(seed) * wsum < wi)
        {
            selected = (int)k;
            phatSel = pci[k];
        }
    }

    LiteReservoir outR = rc;
    float3 visSel = LITE_UNSHADOWED ? (float3)1.0f : rc.tint;
    float3 ySel = yc;
    float3 nySel = nyc;
    LiteLink lSel = lc;
    if (selected >= 0)
    {
        // The winner's payload is only loaded on acceptance.
        const LiteReservoir ri = LiteLoad(g_Reservoirs_current, qpx[selected]);
        outR.s = ri.s;
        visSel = UnpackRGB9E5(visPk[selected]);
        ySel = LiteWorldPosition(ri.s);
        nySel = LiteWorldNormal(ri.s);
        lSel = LiteConnect(rcv, ri.s, ySel, nySel);
    }
    outR.W = (wsum > 0.0f && phatSel > 0.0f) ? LiteSanitizeWeight(wsum / phatSel) : 0.0f;
    const bool shade = outR.W > 0.0f && LiteHasSample(outR.s) && LiteLinkValid(lSel);
    // Unshadowed targets (A/B): the winner's visibility is traced exactly once here.
    if (shade && LITE_UNSHADOWED) visSel = LiteVisibility(rcv, outR.s, lSel, ySel, nySel);

    //====================================
    //SHADING
    //====================================
    // The exact gated broad share times the sample's radiance, geometry,
    // visibility and the contribution weight.
    float3 contribution = 0.0f;
    if (shade)
    {
        float2 iors; uint medium; float3 absorb;
        load_rg_primaryExtra(g_pathStateBuffer, px, iors, medium, absorb);
        contribution = LiteExactBroad(sd, iors, lSel.dir) * outR.s.radiance * lSel.geom * visSel * outR.W;
        if (any(isnan(contribution)) || any(isinf(contribution))) contribution = 0.0f;
    }
    const float4 estimate = gScratchPing[uint3(pixel, 2)];
    gScratchPing[uint3(pixel, 2)] = LITE_DEBUG ? float4(contribution, 0.0f) : estimate + float4(contribution, 0.0f);
}
