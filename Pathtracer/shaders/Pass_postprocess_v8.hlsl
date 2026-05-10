#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//SRGB GAMMA
//====================================
inline float3 sRGBGammaCorrection(float3 color)
{
    float3 result;

    if (color.r <= 0.0031308f)
        result.r = 12.92f * color.r;
    else
        result.r = 1.055f * pow(color.r, 1.0f / 2.4f) - 0.055f;

    if (color.g <= 0.0031308f)
        result.g = 12.92f * color.g;
    else
        result.g = 1.055f * pow(color.g, 1.0f / 2.4f) - 0.055f;

    if (color.b <= 0.0031308f)
        result.b = 12.92f * color.b;
    else
        result.b = 1.055f * pow(color.b, 1.0f / 2.4f) - 0.055f;

    return result;
}

//====================================
//PBR NEUTRAL TONEMAP (kept for reference, not currently called)
//====================================
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

//====================================
//AGX TONEMAP (Troy Sobotka)
//====================================
//"Minimal AgX" form by Benjamin Wrensch (iolite-engine.com).
//Input: scene-linear sRGB-primaries radiance (after exposure scaling).
//Output: sRGB display-encoded values, write straight to the UNORM target.
//DO NOT apply sRGBGammaCorrection after AgX -- the sigmoid + output matrix
//already incorporates the display EOTF.
//Versus PBRNeutral: less vibrant, gentler highlight desaturation, more
//film-like response across wide dynamic range. Default contrast/look ("Base").
static const float3x3 kAgXInputMatrix = float3x3(
    0.842479062253094f,  0.0784335999999992f, 0.0792237451477643f,
    0.0423282422610123f, 0.878468636469772f,  0.0791661274605434f,
    0.0423756549057051f, 0.0784336f,          0.879142973793104f);

static const float3x3 kAgXOutputMatrix = float3x3(
     1.19687900512017f,    -0.0980208811401368f, -0.0990297440797205f,
    -0.0528968517574562f,   1.15190312990417f,   -0.0989611768448433f,
    -0.0529716355144438f,  -0.0980434501171241f,  1.15107367264116f);

//6th-order polynomial fit of the AgX log-domain sigmoid in [0,1]
float3 AgXDefaultContrastApprox(float3 x) {
    float3 x2 = x * x;
    float3 x4 = x2 * x2;
    return  15.5f    * x4 * x2
          - 40.14f   * x4 * x
          + 31.96f   * x4
          -  6.868f  * x2 * x
          +  0.4298f * x2
          +  0.1191f * x
          -  0.00232f;
}

float3 AgX(float3 color) {
    //sRGB -> AgX working primaries
    color = mul(kAgXInputMatrix, color);

    //log2 encode, clamp the exposure range, normalize to [0,1]
    //maxEv bumped from AgX-stock 4.026 -> 6.0 to give bright emitters
    //extra headroom before per-channel clipping desaturates them to white
    const float minEv = -12.47393f;
    const float maxEv =  6.0f;
    color = clamp(log2(max(color, 1e-10f)), minEv, maxEv);
    color = (color - minEv) / (maxEv - minEv);

    //display sigmoid (this IS the EOTF for AgX)
    color = AgXDefaultContrastApprox(color);

    //AgX -> sRGB display primaries
    color = mul(kAgXOutputMatrix, color);

    return saturate(color);
}

//====================================
//INVERSE REINHARD (RECOVER HDR AFTER DLSS RR)
//====================================
//Pass_shading_v8.hlsl applies c / (1 + c) to its DLSS RR input so DLSS sees
//a bounded range (the network mishandles very bright emitters otherwise).
//Postprocess inverts that here so AgX still operates on real HDR — without
//this, emitters get clipped at the shading stage AND compressed again by AgX,
//and end up dimmer than their surroundings.
//Saturating dlssOut bounds the recovered HDR at ~10000, which is well below
//AgX's clipping point at maxEv.
inline float3 InverseDlssReinhard(float3 c) {
    c = saturate(c);
    return c / max(1.0f - c, 1e-4f);
}

//====================================
//EXPOSURE FROM AUTO-EXPOSE STATE
//====================================
//key / exp(smoothed log-lum). Auto-expose finalize already clamped the log-lum
//range, so this can't blow up. Lowered from 0.18 (photographic mid-grey) to
//target a darker mid-tone that reads correctly on screen with AgX.
static const float AE_KEY_VALUE = 0.12f;

float ReadExposure() {
    const float smoothedLogLum = asfloat(gAutoExpose.Load(AE_OFFS_SMOOTHED));
    return AE_KEY_VALUE / max(exp(smoothedLogLum), 1e-6f);
}

//====================================
//POST-PROCESS PASS
//====================================
[numthreads(8, 4, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;

    const float exposure = ReadExposure();

    float3 noisy  = gOutput[uint3(DTid.xy, 0)].xyz;
    //clean came through DLSS RR which received Reinhard-tonemapped input —
    //undo that here so AgX sees the original HDR
    float3 clean  = InverseDlssReinhard(g_dlssOutput[DTid.xy].xyz);
    float3 gt     = gOutput[uint3(DTid.xy, 2)].xyz;
    float3 nrc    = gOutput[uint3(DTid.xy, 3)].xyz;
    float3 refl   = gOutput[uint3(DTid.xy, 4)].xyz;
    float3 albedo = gOutput[uint3(DTid.xy, 5)].xyz;

    //auto-exposure + AgX tonemap on the radiance-bearing slices.
    //AgX outputs sRGB display-encoded values, no extra gamma needed.
    //albedo is linear surface reflectance, gamma-encode it for display.
    noisy = AgX(noisy * exposure);
    clean = AgX(clean * exposure);
    gt    = AgX(gt    * exposure);
    nrc   = AgX(nrc   * exposure);
    refl  = AgX(refl  * exposure);
    albedo = sRGBGammaCorrection(albedo);

    gOutput[uint3(DTid.xy, 0)] = float4(noisy, 0.0f);
    gOutput[uint3(DTid.xy, 1)] = float4(clean, 0.0f);
    gOutput[uint3(DTid.xy, 2)] = float4(gt, 0.0f);
    gOutput[uint3(DTid.xy, 3)] = float4(nrc, 0.0f);
    gOutput[uint3(DTid.xy, 4)] = float4(refl, 0.0f);
    gOutput[uint3(DTid.xy, 5)] = float4(albedo, 0.0f);
}
