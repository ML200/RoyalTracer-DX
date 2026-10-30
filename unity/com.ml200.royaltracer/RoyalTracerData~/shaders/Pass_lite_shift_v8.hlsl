#include "Includes_v8.hlsli"
#include "RestirLite_v8.hlsli"

// Spatial reuse, first half: partner targets and visibility at this receiver.
[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    const uint2 pixel = tid.xy;
    const uint2 dims = uint2(IMG_W, IMG_H);
    const uint px = MapPixelID(dims, (int2)pixel);
    if (load_flagsWord(g_sample_current, px) & SD_FLAG_NOBOUNCE) return;
    const LiteReservoir rc = LiteLoad(g_liteReservoirs, px);
    if (rc.M == 0u) return;
    const SDRecord sd = load_SD(g_sample_current, px);
    const LiteReceiver rcv = LiteReceiverFromSD(sd);
    const uint   matID = sd.matID;
    const float3 cam = InitOrigin();
    const float  camDist = length(sd.x1 - cam);

    uint   knownInst[LITE_SLOTS_MAX + 1u];
    float3 knownPos[LITE_SLOTS_MAX + 1u];
    uint   knownVisPk[LITE_SLOTS_MAX + 1u];
    float  knownVisLuma[LITE_SLOTS_MAX + 1u];
    uint   known = 0u;
    float  own = 0.0f;
    if (LiteHasSample(rc.s))
    {
        const float3 y  = LiteWorldPosition(rc.s);
        const float3 ny = LiteWorldNormal(rc.s);
        const float3 vis = LITE_UNSHADOWED ? (float3)1.0f : rc.tint;
        own = LiteTarget(rcv.albedo, rc.s, LiteConnect(rcv, rc.s, y, ny), y, Luma(vis));
        knownInst[0] = rc.s.instance;
        knownPos[0] = rc.s.position;
        knownVisPk[0] = PackRGB9E5(vis);
        knownVisLuma[0] = Luma(vis);
        known = 1u;
    }

    uint mask = 0u;
    const uint slots = LITE_SPATIAL ? min(lite_spatSlots, LITE_SLOTS_MAX) : 0u;
    [loop]
    for (uint s = 0u; s < slots; ++s)
    {
        const int2 q = int2(pixel) + LiteReuseDelta(pixel, s);
        if (any(q < 0) || any(q >= (int2)dims)) continue;
        const uint qpx = MapPixelID(dims, q);
        if (load_flagsWord(g_sample_current, qpx) & SD_FLAG_NOBOUNCE) continue;
        const LiteReservoir ri = LiteLoad(g_liteReservoirs, qpx);
        if (ri.M == 0u) continue;
        const SDRecord sdi = load_SD(g_sample_current, qpx);
        if (sdi.matID != matID) continue;
        if (!LiteSimilar(rcv, LiteReceiverFromSD(sdi), camDist, length(sdi.x1 - cam))) continue;
        mask |= 1u << s;

        float phat  = 0.0f;
        uint  visPk = 0u;
        if (LiteHasSample(ri.s))
        {
            const float3 y  = LiteWorldPosition(ri.s);
            const float3 ny = LiteWorldNormal(ri.s);
            const LiteLink l = LiteConnect(rcv, ri.s, y, ny);
            if (LiteLinkValid(l))
            {
                // Target before visibility: the only value live across the trace.
                const float phatDry = Luma(rcv.albedo * ri.s.radiance) * LITE_INV_PI * l.geom;
                float visLuma = 0.0f;
                bool  found = false;
                [unroll]
                for (uint k = 0u; k < LITE_SLOTS_MAX + 1u; ++k)
                {
                    if (k < known && !found && knownInst[k] == ri.s.instance && all(knownPos[k] == ri.s.position))
                    {
                        visPk = knownVisPk[k];
                        visLuma = knownVisLuma[k];
                        found = true;
                    }
                }
                if (!found)
                {
                    const float3 vis = LITE_UNSHADOWED ? (float3)1.0f : LiteVisibility(rcv, ri.s, l, y, ny);
                    visPk = PackRGB9E5(vis);
                    visLuma = Luma(vis);
                    [unroll]
                    for (uint k = 0u; k < LITE_SLOTS_MAX + 1u; ++k)
                    {
                        if (k != known) continue;
                        knownInst[k] = ri.s.instance;
                        knownPos[k] = ri.s.position;
                        knownVisPk[k] = visPk;
                        knownVisLuma[k] = visLuma;
                    }
                    if (known < LITE_SLOTS_MAX + 1u) ++known;
                }
                phat = visLuma > 0.0f ? phatDry * visLuma : 0.0f;
            }
        }
        g_pathStateBuffer.Store3(LiteShiftAddress(px, s), uint3(asuint(phat), visPk, qpx));
    }
    g_pathStateBuffer.Store2(LiteShiftMaskAddress(px), uint2(mask, asuint(own)));
}
