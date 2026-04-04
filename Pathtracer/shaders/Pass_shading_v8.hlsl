#include "Includes_v8.hlsli"


// ─────────────────────────────────────────────────────────────────────────────
//  SHADING PASS
// ─────────────────────────────────────────────────────────────────────────────
[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;
    gOutput[uint3(DTid.xy, 0)] = float4(0, 0, 0, 0);

    float3 output_DI  = gScratchPing[uint3(DTid.xy, 1)];
    float3 output_GI  = gScratchPing[uint3(DTid.xy, 2)];
    float3 sunDirect  = gScratchPing[uint3(DTid.xy, 3)].rgb;

    float3 accumulation = output_DI + output_GI + sunDirect;

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

    bool isEmissiveOrSky = any(sdata.L1 > 0.0f);
    if (isEmissiveOrSky)
    {
        // Emitters have a valid surface position (x1); sky does not (x1 == 0)
        bool hasPosition = dot(sdata.x1, sdata.x1) > 0.0f;

        if (hasPosition)
        {
            // Emitter surface: compute depth + motion vectors like regular geometry
            g_dlssDepth[DTid.xy] = DLSS_LinearDepthFromWorldPos(sdata.x1);

            float2 curPix = DTid.xy;
            float2 prevPix = GetLastFramePixelCoordinates_Float(
                sdata.x1, prevView, prevProjection, dims, sdata.objID) - jitter;
            bool validPrev = (prevPix.x >= 0 && prevPix.y >= 0 &&
                              prevPix.x < IMG_W && prevPix.y < IMG_H);
            g_dlssMVec[curPix] = validPrev ? float2(prevPix - curPix) : float2(0, 0);
        }
        else
        {
            // Sky: no surface, infinite depth, no motion
            g_dlssDepth[DTid.xy] = 65504.0f;
            g_dlssMVec[DTid.xy] = float2(0.0f, 0.0f);
        }

        g_dlssNormals[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        g_dlssSpecularAlbedo[DTid.xy] = float4(0.5f, 0.5f, 0.5f, 0.0f);
        g_dlssDiffuseAlbedo[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        g_dlssRoughness[DTid.xy] = 0.0f;
        g_dlssSpecHitDist[DTid.xy] = hasPosition ? 0.0f : 65504.0f;

        // Tonemap bright emitters before DLSS-RR to prevent ghosting/ringing
        float3 dlssColor = accumulation / (1.0f + accumulation);
        g_dlssInput[DTid.xy] = float4(saturate(dlssColor), 1.0f);
    }
    else{
        // DLSS RR input data:
        g_dlssDepth[DTid.xy] = DLSS_LinearDepthFromWorldPos(sdata.x1);

        g_dlssNormals[DTid.xy] = float4(sdata.n1_s, 0.0f);
        // Refetch material for DLSS albedo/roughness
        float3 shadingKd; float shadingPr, shadingPm;
        RefetchMaterial(sdata.matID, sdata.uv, shadingKd, shadingPr, shadingPm);
        g_dlssDiffuseAlbedo[DTid.xy] = float4(shadingKd, 1.0f);
        g_dlssRoughness[DTid.xy] = shadingPr;

        float2 curPix = DTid.xy;
        float2 prevPix = GetLastFramePixelCoordinates_Float(sdata.x1, prevView, prevProjection, dims, sdata.objID) - jitter;

        bool validPrev = (prevPix.x >= 0 && prevPix.y >= 0 && prevPix.x < IMG_W && prevPix.y < IMG_H);

        float2 mvPixels = validPrev ? float2(prevPix - curPix) : float2(0.0, 0.0);

        g_dlssMVec[curPix] = mvPixels;
        gOutput[uint3(DTid.xy, 10)] = float4(abs(mvPixels), 0.0f, 1.0f);

        // Specular albedo
        float3 specularAlbedo = EnvBRDFApprox2(shadingKd, shadingPr, shadingPm, dot(normalize(sdata.x1 - mul(viewI, float4(0, 0, 0, 1)).xyz), sdata.n1_s));
        g_dlssSpecularAlbedo[DTid.xy] = float4(specularAlbedo, 0.0f);

        // Post-spatial GI reservoir for stable hit distance
        Reservoir_GI rdi = loadReservoirGI(g_Reservoirs_last_gi, pixelIdx);
        g_dlssSpecHitDist[DTid.xy] = length(rdi.x2_gi - sdata.x1);

        g_dlssInput[DTid.xy] = float4(accumulation, 1.0f);
    }
}
