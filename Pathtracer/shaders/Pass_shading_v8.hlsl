#include "Includes_v8.hlsli"

inline float3 sRGBGammaCorrection(float3 color)
{
    float3 result;

    // Red channel
    if (color.r <= 0.0031308f)
        result.r = 12.92f * color.r;
    else
        result.r = 1.055f * pow(color.r, 1.0f / 2.4f) - 0.055f;

    // Green channel
    if (color.g <= 0.0031308f)
        result.g = 12.92f * color.g;
    else
        result.g = 1.055f * pow(color.g, 1.0f / 2.4f) - 0.055f;

    // Blue channel
    if (color.b <= 0.0031308f)
        result.b = 12.92f * color.b;
    else
        result.b = 1.055f * pow(color.b, 1.0f / 2.4f) - 0.055f;

    return result;
}

float3 PBRNeutral(float3 color) {
    const float startCompression = 0.8f - 0.04f;
    const float desaturation = 0.15f;

    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08f ? x - 6.25f * x * x : 0.04f;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return max(color, 0.0f);

    float d = 1.0f - startCompression;
    float newPeak = 1.0f - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1.0f - 1.0f / (desaturation * (peak - newPeak) + 1.0f);
    return lerp(color, newPeak.xxx, g);
}

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

    float3 accumulation = output_DI + output_GI;
    float3 gt = gScratchPing[uint3(DTid.xy, 3)];

    /*bool cameraChanged = false;
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        if (any(view[i] != prevView[i])) cameraChanged = true;
    }
    static const float MAX_SAMPLES     = 1000000.0;

    float4 prev        = gPermanentData[DTid.xy];   // rgb = running avg, a = N
    float3 prevAvg     = prev.rgb;
    float  prevSamples = prev.a;

    float3 newAvg;
    float  newSamples;
    if (cameraChanged)
    {
        // Camera moved: reset running average and sample count
        newAvg     = gt;
        newSamples = 1.0h;
    }
    else
    {
        newSamples = min(prevSamples + 1.0h, MAX_SAMPLES);
        float invN  = 1.0h / newSamples;
        newAvg     = mad(gt - prevAvg, invN, prevAvg);
    }

    // store back
    gPermanentData[DTid.xy] = float4(newAvg, newSamples);*/

    float3 sceneLinear = accumulation;
    //float3 outSRGB     = sRGBGammaCorrection(saturate(sceneLinear));

    // DLSS RR input data:
    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, DTid.xy);
    SampleData sdata = loadSampleData(g_sample_current, pixelIdx);

    gOutput[uint3(DTid.xy, 0)] = float4(accumulation, 1.0f);


    g_dlssDepth[DTid.xy] = length(sdata.x1 - mul(viewI, float4(0, 0, 0, 1)).xyz);

    g_dlssNormals[DTid.xy] = float4(sdata.n1_s, 0.0f);
    g_dlssDiffuseAlbedo[DTid.xy] = float4(sdata.localKd, 1.0f);
    g_dlssRoughness[DTid.xy] = sdata.localPr;

    // ALWAYS write MV
    float2 curPix = DTid.xy;
    float2 prevPix = GetLastFramePixelCoordinates_Float(sdata.x1, prevView, prevProjection, dims, sdata.objID);

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

    // Denoiser__________________________________________________
    // Slice 0: Raw noisy signal
    /*gScratchPing[uint3(DTid.xy, 0)] = float4(accumulation, 1.0f);

    // Slice 2: Albedo (base color)
    gScratchPing[uint3(DTid.xy, 2)] = float4(sdata.localKd, 1.0f);

    // Slice 3: x = isEmissive, y = Roughness, z = ObjID
    // Note: If you have an emissive material check in sdata (e.g. sdata.emission),
    // set this to 1.0f so the denoiser skips blurring your lights. Defaulting to 0.0f.
    float isEmissive =  any(sdata.L1 > 0.0f)?1.0f:0.0f;
    gScratchPing[uint3(DTid.xy, 3)] = float4(isEmissive, sdata.localPr, asfloat((uint)sdata.objID), 0.0f);

    // Slice 4: World-Space Normal
    gScratchPing[uint3(DTid.xy, 4)] = float4(sdata.n1_s, 0.0f);

    // Slice 5: xyz = World-Space Pos, w = current sample weight (M_cur)
    gScratchPing[uint3(DTid.xy, 5)] = float4(sdata.x1, 1.0f);*/



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
