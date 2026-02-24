#include "Includes_v8.hlsli"

[numthreads(8, 4, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;
    gDispatchIdx = uint3(DTid.xy, 0);

    uint  pixelIdx = MapPixelID(float2(gImageWidth, gImageHeight), DTid.xy);
    SampleData cur = loadSampleData(g_sample_current, pixelIdx);

    // Store SampleData temporally
    storeSampleData(g_sample_last, pixelIdx, cur);

    /*Defined as (0 is general render output) (first number current frame, last number previous frame):
    - 2/7: albedo
    - 3/8: emission, roughness, objID
    - 4/9: normal
    - 5/10: position_curr, M_curr
    - 6/11: position_last
    */
    //update the last data
    gScratchPing[uint3(DTid.xy, 7)] = gScratchPing[uint3(DTid.xy, 2)];
    gScratchPing[uint3(DTid.xy, 8)] = gScratchPing[uint3(DTid.xy, 3)];
    gScratchPing[uint3(DTid.xy, 9)] = gScratchPing[uint3(DTid.xy, 4)];
    gScratchPing[uint3(DTid.xy, 10)] = gScratchPing[uint3(DTid.xy, 5)];
    gScratchPing[uint3(DTid.xy, 11)] = gScratchPing[uint3(DTid.xy, 6)];

    // Post Processing and image write
    float3 finalColor = gScratchPing[uint3(DTid.xy, 1)].xyz;

    // Store temporal data for reprojection
    gPermanentData[DTid.xy] = gScratchPing[uint3(DTid.xy,12)];
    gOutput[uint3(DTid.xy, 1)] = float4(finalColor, 1);
}
