#define COMPUTE_PASS
#define SPMIS_GRID_NONCOHERENT
#include "Includes_v8.hlsli"
#include "RestirLite_v8.hlsli"

//====================================
//PAIRED SPATIAL SHIFT
//====================================
// Every pixel evaluates its own target at each valid partner's sample and
// parks it, with the partner's pixel index. The pairing is self-inverting,
// so the merge pass finds the reverse evaluation in the partner's slot of
// the same index: one ray per pair instead of two. A partner holding the
// same point as this pixel, or as an earlier partner, reuses the visibility
// this receiver already established for it (no ray).
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;
    const uint2 pixel = tid.xy;
    const uint2 dims = uint2(IMG_W, IMG_H);
    const uint px = MapPixelID(dims, (int2)pixel);
    if (load_flagsWord(g_sample_current, px) & SD_FLAG_NOBOUNCE) return;
    const LiteReservoir rc = LiteLoad(g_Reservoirs_current, px);
    if (rc.M == 0u) return;
    const SDRecord sd = load_SD(g_sample_current, px);
    const LiteReceiver rcv = LiteReceiverFromSD(sd);
    const float3 cam = InitOrigin();
    const float camDist = length(sd.x1 - cam);

    // Own target of the own sample: partners read it as their p_hat_i(y_i).
    // The own visibility tint seeds the list of points already evaluated.
    LiteSample knownS[LITE_SLOTS_MAX + 1u];
    float3     knownVis[LITE_SLOTS_MAX + 1u];
    uint       known = 0u;
    float own = 0.0f;
    if (LiteHasSample(rc.s))
    {
        const float3 y = LiteWorldPosition(rc.s);
        const float3 ny = LiteWorldNormal(rc.s);
        const float3 vis = LITE_UNSHADOWED ? (float3)1.0f : rc.tint;
        own = LiteTarget(rcv.albedo, rc.s, LiteConnect(rcv, rc.s, y, ny), y, Luma(vis));
        knownS[0] = rc.s;
        knownVis[0] = vis;
        known = 1u;
    }

    uint mask = 0u;
    const uint slots = LITE_SPATIAL ? min(lite_spatSlots, LITE_SLOTS_MAX) : 0u;
    [loop]
    for (uint s = 0u; s < slots; ++s)
    {
        // Every test below is symmetric, so the partner reaches the same
        // verdict about this pixel in its slot s.
        const int2 q = int2(pixel) + LiteReuseDelta(pixel, s);
        if (any(q < 0) || any(q >= (int2)dims)) continue;
        const uint qpx = MapPixelID(dims, q);
        if (load_flagsWord(g_sample_current, qpx) & SD_FLAG_NOBOUNCE) continue;
        const LiteReservoir ri = LiteLoad(g_Reservoirs_current, qpx);
        if (ri.M == 0u) continue;
        const SDRecord sdi = load_SD(g_sample_current, qpx);
        if (sdi.matID != sd.matID) continue;
        const LiteReceiver rcvI = LiteReceiverFromSD(sdi);
        if (!LiteSimilar(rcv, rcvI, camDist, length(sdi.x1 - cam))) continue;
        mask |= 1u << s;

        float phat = 0.0f;
        float3 vis = 0.0f;
        if (LiteHasSample(ri.s))
        {
            const float3 y = LiteWorldPosition(ri.s);
            const float3 ny = LiteWorldNormal(ri.s);
            const LiteLink l = LiteConnect(rcv, ri.s, y, ny);
            if (LiteLinkValid(l))
            {
                bool found = false;
                [unroll]
                for (uint k = 0u; k < LITE_SLOTS_MAX + 1u; ++k)
                {
                    if (k < known && !found && LiteSameSample(knownS[k], ri.s))
                    {
                        vis = knownVis[k];
                        found = true;
                    }
                }
                if (!found)
                {
                    vis = LITE_UNSHADOWED ? (float3)1.0f : LiteVisibility(rcv, ri.s, l, y, ny);
                    if (known < LITE_SLOTS_MAX + 1u)
                    {
                        knownS[known] = ri.s;
                        knownVis[known] = vis;
                        ++known;
                    }
                }
                phat = LiteTarget(rcv.albedo, ri.s, l, y, Luma(vis));
            }
        }
        g_pathStateBuffer.Store3(LiteShiftAddress(px, s), uint3(asuint(phat), PackRGB9E5(vis), qpx));
    }
    g_pathStateBuffer.Store2(LiteShiftMaskAddress(px), uint2(mask, asuint(own)));
}
