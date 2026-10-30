#include "Includes_v8.hlsli"
#include "PtNee_v8.hlsli"

// The light samples the trace pass queued for in-place vertices after the deferred one (where the cache
// refused): learned light-tree pick, sun, shadow rays and BSDF, added into the path's relL and reuse
// record with the weights the path carried there. A pass of its own, so the trace pipeline holds none of
// this code or its registers, and each wave runs one kind of work.
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_GroupID, uint gi : SV_GroupIndex)
{
    // 256 consecutive slots per group: a wave takes requests one material-shader wave queued together.
    const uint slot = (gid.y * ((IMG_W + 15u) / 16u) + gid.x) * 256u + gi;
    if (slot >= PtNeeQueueCount()) return;

    PtNeeIO p;
    uint2  pixel;
    float3 weight, liteWeight;
    PtNeeQueueLoad(slot, p, pixel, weight, liteWeight);

    float3 direct, liteDirect;
    PtNeeEvaluate(p, pixel, false, direct, liteDirect);

    const uint pixelIdx = MapPixelID(uint2(IMG_W, IMG_H), pixel);
    DvAtomicAddPathRelL(pixelIdx, weight * direct);
    if (any(liteWeight > 0.0f)) DvAtomicAddLiteL(pixelIdx, liteWeight * liteDirect);
}
