#include "Includes_v7.hlsli"

//─────────────────────────────────────────────────────────────────────────────
//  Initial Sampling DI
//─────────────────────────────────────────────────────────────────────────────
[numthreads(8, 8, 1)]
void main(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x >= IMG_W || tid.y >= IMG_H) return;
    gDispatchIdx = tid;

    uint2  launchIndex   = tid.xy;
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, launchIndex);

    // Trace the initial camera rays and store the sdata.
    SampleData sdata = SampleCameraRay(pixelIdx, launchIndex, dims);

    //Debug
    //gScratchPing[uint3(tid.xy, 1)] = float4((sdata.n1+1.0f)/2.0f, 0.0f); // Norm
    //gScratchPing[uint3(tid.xy, 1)] = float4(sdata.x1, 0.0f); // Pos
    gScratchPing[uint3(tid.xy, 1)] = float4(materials[sdata.matID].Kd.xyz, 0.0f); // Material

}