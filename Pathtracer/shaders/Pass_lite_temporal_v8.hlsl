#define COMPUTE_PASS
#define SPMIS_GRID_NONCOHERENT
#include "Includes_v8.hlsli"
#include "RestirLite_v8.hlsli"

//====================================
//TEMPORAL REUSE
//====================================
// Merges the reprojected (and permuted) previous final reservoir into this
// frame's candidate reservoir, in place. Both targets carry visibility, so
// the two cross evaluations trace one shadow ray each: the history's
// surface toward the candidate, this surface toward the history's sample.
// When both hold the same point, both visibilities are already known.
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
    if (rc.M == 0u || !LITE_TEMPORAL) return;

    const SDRecord sd = load_SD(g_sample_current, px);
    const LiteReceiver rcv = LiteReceiverFromSD(sd);
    int2 prev = GetBestReprojectedPixel_d(sd.x1, prevView, prevProjection, float2(dims), sd.instID);
    // Off screen or behind the camera (the leading border under motion): the
    // deprecated pipeline's fallback reads the history at this pixel's own
    // position instead. The geometry test below still decides, and the MIS
    // weighs whatever surface that history belongs to, so nothing is lost
    // but the band of pixels whose own candidates all came up empty.
    if (prev.x < 0) prev = (int2)pixel;
    if (LITE_PERMUTE)
    {
        // The deprecated pipeline's permutation sampling: a 4x4 shuffle of the
        // history source, new every frame, so no pixel keeps its own chain.
        // A shuffle that leaves the screen keeps the unshuffled source.
        int2 permuted = prev;
        ApplyPermutationSampling(permuted, Hash32(sharc_frame * 0x9E3779B9u ^ 0x5045524du) & 15u);
        if (all(permuted >= 0) && all(permuted < (int2)dims)) prev = permuted;
    }
    const uint tpx = MapPixelID(dims, prev);
    if (load_flagsWord(g_sample_last, tpx) & SD_FLAG_NOBOUNCE) return;
    const LiteReservoir rt = LiteLoad(g_Reservoirs_last, tpx);
    if (rt.M == 0u) return;
    const SDRecord sdt = load_SD(g_sample_last, tpx);
    if (sdt.matID != sd.matID) return;
    const LiteReceiver rcvT = LiteReceiverFromSD(sdt);
    const float3 cam = InitOrigin();
    if (!LiteSimilar(rcv, rcvT, length(sd.x1 - cam), length(sdt.x1 - cam))) return;

    // History confidence: the cap, collapsed toward one where the
    // duplication map says this sample already fills the neighbourhood.
    float Mt = min((float)rt.M, (float)lite_tempMcap);
    if (LITE_DUPMAP)
    {
        const float D = saturate(gScratchPing[uint3(prev, LITE_DUP_SCRATCH)].x);
        Mt = min(Mt, lerp((float)lite_tempMcap, 1.0f, pow(D, rs_corrReductionPow)));
    }
    const float Mc = (float)rc.M;
    const float vcc = LITE_UNSHADOWED ? 1.0f : Luma(rc.tint); // this surface's visibility of its sample
    const float vtt = LITE_UNSHADOWED ? 1.0f : Luma(rt.tint); // the history surface's, of its sample

    // p_ab: target of surface a at the sample of b (c = current, t = history)
    float pcc = 0.0f, ptc = 0.0f, pct = 0.0f, ptt = 0.0f;
    float3 visCT = 0.0f;
    const bool hasC = LiteHasSample(rc.s);
    const bool hasT = LiteHasSample(rt.s);
    if (hasC && hasT && LiteSameSample(rc.s, rt.s))
    {
        // Same point: each surface already knows its visibility of it.
        const float3 y = LiteWorldPosition(rc.s);
        const float3 ny = LiteWorldNormal(rc.s);
        const LiteLink lc = LiteConnect(rcv, rc.s, y, ny);
        const LiteLink lt = LiteConnect(rcvT, rc.s, y, ny);
        pcc = LiteTarget(rcv.albedo, rc.s, lc, y, vcc);
        ptc = LiteTarget(rcvT.albedo, rc.s, lt, y, vtt);
        pct = LiteTarget(rcv.albedo, rt.s, lc, y, vcc);
        ptt = LiteTarget(rcvT.albedo, rt.s, lt, y, vtt);
        visCT = LITE_UNSHADOWED ? (float3)1.0f : rc.tint;
    }
    else
    {
        if (hasC)
        {
            const float3 y = LiteWorldPosition(rc.s);
            const float3 ny = LiteWorldNormal(rc.s);
            const LiteLink lc = LiteConnect(rcv, rc.s, y, ny);
            pcc = LiteTarget(rcv.albedo, rc.s, lc, y, vcc);
            const LiteLink lt = LiteConnect(rcvT, rc.s, y, ny);
            float vt = 1.0f;
            if (!LITE_UNSHADOWED && LiteLinkValid(lt)) vt = Luma(LiteVisibility(rcvT, rc.s, lt, y, ny));
            ptc = LiteTarget(rcvT.albedo, rc.s, lt, y, vt);
        }
        if (hasT)
        {
            const float3 y = LiteWorldPosition(rt.s);
            const float3 ny = LiteWorldNormal(rt.s);
            const LiteLink lt = LiteConnect(rcvT, rt.s, y, ny);
            ptt = LiteTarget(rcvT.albedo, rt.s, lt, y, vtt);
            const LiteLink lc = LiteConnect(rcv, rt.s, y, ny);
            if (LiteLinkValid(lc))
            {
                visCT = LITE_UNSHADOWED ? (float3)1.0f : LiteVisibility(rcv, rt.s, lc, y, ny);
                pct = LiteTarget(rcv.albedo, rt.s, lc, y, Luma(visCT));
            }
        }
    }

    const float mc = LiteBalance(Mc, pcc, Mt, ptc);
    const float mt = LiteBalance(Mt, ptt, Mc, pct);
    const float wc = mc * pcc * rc.W;
    const float wt = mt * pct * rt.W;
    const float wsum = wc + wt;
    float phatSel = pcc;
    uint seed = GetSeed(pixel, sharc_frame, 0x54454d50u).x;
    if (wt > 0.0f && RandomFloatPCG(seed) * wsum < wt)
    {
        rc.s = rt.s;
        rc.tint = visCT;
        phatSel = pct;
    }
    rc.M = min((uint)round(Mc + Mt), 255u);
    rc.W = (wsum > 0.0f && phatSel > 0.0f) ? LiteClampW(wsum / phatSel) : 0.0f;
    LiteStore(g_Reservoirs_current, px, rc);
}
