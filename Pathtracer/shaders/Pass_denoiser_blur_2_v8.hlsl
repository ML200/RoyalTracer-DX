#include "Includes_v8.hlsli"

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    gDispatchIdx = DTid;
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;

    uint2  launch = DTid.xy;
    uint2  dimsI  = DispatchRaysDimensions().xy;
    float2 dims   = float2(dimsI);
    uint   pIdx   = MapPixelID(dims, launch);

    // Blur kernel
    float3 output = AtrousKernel(launch, 4, 1);

    // Store accumulated result
    gScratchPing[uint3(launch, 0)] = float4(output, 1);
}


