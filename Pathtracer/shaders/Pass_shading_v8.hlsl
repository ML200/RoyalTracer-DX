#include "Includes_v8.hlsli"


// ─────────────────────────────────────────────────────────────────────────────
//  SHADING PASS
// ─────────────────────────────────────────────────────────────────────────────
[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;
    gOutput[uint3(DTid.xy, 0)] = float4(0, 0, 0, 0);

    float3 output_DI = gScratchPing[uint3(DTid.xy, 1)];
    float3 output_GI = gScratchPing[uint3(DTid.xy, 2)];

    float3 accumulation = /*output_DI +*/ output_GI;
    //float3 gt = gScratchPing[uint3(DTid.xy, 3)];

    bool cameraChanged = false;
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        if (any(view[i] != prevView[i])) cameraChanged = true;
    }
    static const float MAX_SAMPLES     = 1000.0;

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
    gOutput[uint3(DTid.xy, 2)] = float4(newAvg, 1.0f);

    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, DTid.xy);
    SampleData sdata = loadSampleData(g_sample_current, pixelIdx);

    gOutput[uint3(DTid.xy, 0)] = float4(accumulation, 1.0f);

    bool isSky = any(sdata.L1 > 0.0f);
    if (isSky)
    {
        g_dlssDepth[DTid.xy] = 65504.0f;

        // GUIDE SECTION 3.4.3: Sky has no surface orientation.
        g_dlssNormals[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);

        // GUIDE SECTION 3.4.2: Explicitly recommends (0.5) for Sky Specular Albedo
        g_dlssSpecularAlbedo[DTid.xy] = float4(0.5f, 0.5f, 0.5f, 0.0f);

        // Sky is perfect emission, no diffuse shading.
        g_dlssDiffuseAlbedo[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);

        // Sky has no roughness.
        g_dlssRoughness[DTid.xy] = 0.0f;

        // GUIDE SECTION 3.4.9: Specular Hit Distance.
        // Since sky is "infinite" or has no primary surface hit, use a safe max value.
        g_dlssSpecHitDist[DTid.xy] = 65504.0f; // FP16 Max

        // Optional: Sky motion should usually be 0 or calculated via camera rotation only.
        g_dlssMVec[DTid.xy] = float2(0.0f, 0.0f);

        g_dlssInput[DTid.xy] = float4(accumulation, 1.0f);
    }
    else{
        // DLSS RR input data:
        g_dlssDepth[DTid.xy] = DLSS_LinearDepthFromWorldPos(sdata.x1);

        g_dlssNormals[DTid.xy] = float4(sdata.n1_s, 0.0f);
        g_dlssDiffuseAlbedo[DTid.xy] = float4(sdata.localKd, 1.0f);
        g_dlssRoughness[DTid.xy] = sdata.localPr;

        // ALWAYS write MV
        float2 curPix = DTid.xy;
        float2 prevPix = GetLastFramePixelCoordinates_Float(sdata.x1, prevView, prevProjection, dims, sdata.objID) - jitter;

        // Clamp or invalidate if outside
        bool validPrev = (prevPix.x >= 0 && prevPix.y >= 0 && prevPix.x < IMG_W && prevPix.y < IMG_H);

        // DLSS expects “current pixel maps to previous frame position”
        float2 mvPixels = validPrev ? float2(prevPix - curPix) : float2(0.0, 0.0);

        g_dlssMVec[curPix] = mvPixels;
        gOutput[uint3(DTid.xy, 10)] = float4(abs(mvPixels),0.0f, 1.0f);

        // Specular albedo
        float3 specularAlbedo = EnvBRDFApprox2(sdata.localKd, sdata.localPr, sdata.localPm, dot(normalize(sdata.x1 - mul(viewI, float4(0, 0, 0, 1)).xyz), sdata.n1_s));
        g_dlssSpecularAlbedo[DTid.xy] = float4(specularAlbedo, 0.0f);

        Reservoir_GI rdi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);
        g_dlssSpecHitDist[DTid.xy] = length(rdi.x2_gi - sdata.x1);

        g_dlssInput[DTid.xy] = float4(accumulation, 1.0f);
    }


    /*float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, DTid.xy);
    SampleData sdata = loadSampleData(g_sample_current, pixelIdx);
    gScratchPing[uint3(DTid.xy, 7)].x = GetPHat(output_GI);
    gScratchPing[uint3(DTid.xy, 7)].y = GetPHat(newAvg);
    gScratchPing[uint3(DTid.xy, 7)].z = GetPHat(gt);
    gScratchPing[uint3(DTid.xy, 7)].w = GetPHat(sdata.localKd);

    float depthVal = length(sdata.x1 - mul(viewI, float4(0, 0, 0, 1)).xyz);
    Reservoir_GI rdi = loadReservoirGI(g_Reservoirs_current_gi, pixelIdx);
    gScratchPing[uint3(DTid.xy, 8)] = float4(rdi.localPr_gi, length(rdi.x2_gi - sdata.x1), 0.0f, 0.0f);
    gScratchPing[uint3(DTid.xy, 9)] = float4(sdata.n1_s, 0.0f);

    gOutput[uint3(DTid.xy, 10)] = gScratchPing[uint3(DTid.xy, 7)].x;
    gOutput[uint3(DTid.xy, 11)] = gScratchPing[uint3(DTid.xy, 7)].y;*/
}
