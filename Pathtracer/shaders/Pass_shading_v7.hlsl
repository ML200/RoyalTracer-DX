#include "Includes_v7.hlsli"

// ─────────────────────────────────────────────────────────────────────────────
//  SHADING PASS
// ─────────────────────────────────────────────────────────────────────────────
[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;

    // Load the DI pipeline output
    float3 output_DI = gScratchPing[uint3(DTid.xy, 1)];
    // Load the GI pipeline output
    float3 output_GI = gScratchPing[uint3(DTid.xy, 2)];

    float3 accumulation = output_DI + output_GI;

    bool cameraChanged = false;
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        if (any(view[i] != prevView[i])) cameraChanged = true;
    }
    static const float MAX_SAMPLES     = 100000.0h;  // tune to taste

    float4 prev        = gPermanentData[DTid.xy];   // rgb = running avg, a = N
    float3 prevAvg     = prev.rgb;
    float  prevSamples = prev.a;

    float3 newAvg;
    float  newSamples;
    if (cameraChanged)
    {
        // Camera moved: reset running average and sample count
        newAvg     = accumulation;
        newSamples = 1.0h;
    }
    else
    {
        newSamples = min(prevSamples + 1.0h, MAX_SAMPLES);
        float invN  = 1.0h / newSamples;
        newAvg     = mad(accumulation - prevAvg, invN, prevAvg);
    }

    // store back
    gPermanentData[DTid.xy] = float4(newAvg, newSamples);

    // show accumulated image
    float3 fColor = sRGBGammaCorrection(newAvg);
    gOutput[uint3(DTid.xy, 0)]  = float4(fColor, 1);
    //override if target is real time undenoised
    float3 finalColor = sRGBGammaCorrection(accumulation);
    //gOutput[uint3(DTid.xy, 0)]  = float4(finalColor, 1);

    // Denoiser buffers etc.
    /*uint2  launchIndex   = DTid.xy;
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, launchIndex);
    SampleData sdata = loadSampleData(g_sample_current, pixelIdx);
    SampleData sdata_l = loadSampleData(g_sample_last, pixelIdx);

    gScratchPing[uint3(DTid.xy, 2)] = length(materials[sdata.matID].Ke) > 0.0f ? float4(materials[sdata.matID].Ke, 0) : materials[sdata.matID].Kd;
    gScratchPing[uint3(DTid.xy, 3)] = float4(length(materials[sdata.matID].Ke) > 0.0f ? 1 : 0, materials[sdata.matID].Pr_Pm_Ps_Pc.x, sdata.objID, 0);
    gScratchPing[uint3(DTid.xy, 4)] = float4(sdata.n1,0);

    Reservoir_GI rgi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);
    gScratchPing[uint3(DTid.xy, 5)] = float4(sdata.x1, rgi.M_gi);
    gScratchPing[uint3(DTid.xy, 6)] = float4(sdata_l.x1,0);

    gScratchPing[uint3(DTid.xy, 0)] = float4(accumulation, 0);*/
}
