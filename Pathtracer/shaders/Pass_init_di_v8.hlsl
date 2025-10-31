#include "Includes_v8.hlsli"

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
    // Seeding
    RandomData rdata = initRandomData(launchIndex, uint2(8,8), time, 1u);

    // Trace the initial camera rays and store the sdata.
    SampleData sdata = SampleCameraRay(pixelIdx, launchIndex, dims);

    // Create the path state +  throughput state object from the sdata information
    PathState pstate = initPathState(sdata.x1, sdata.n1, sdata.o, sdata.objID, sdata.matID);
    ThroughputState tstate = initThroughputState();


    //Debug
    gScratchPing[uint3(tid.xy, 1)] = float4((sdata.n1+1.0f)/2.0f, 0.0f); // Norm
    //gScratchPing[uint3(tid.xy, 1)] = float4(sdata.x1, 0.0f); // Pos
    //gScratchPing[uint3(tid.xy, 1)] = float4(materials[sdata.matID].Kd.xyz, 0.0f); // Material
    //gScratchPing[uint3(tid.xy, 1)] = float4(sdata.L1, 0.0f); // Emission

}