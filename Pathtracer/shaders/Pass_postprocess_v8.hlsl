#include "Includes_v8.hlsli"

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

static const float3x3 kAgXInputMatrix = float3x3(
    0.842479062253094f,  0.0784335999999992f, 0.0792237451477643f,
    0.0423282422610123f, 0.878468636469772f,  0.0791661274605434f,
    0.0423756549057051f, 0.0784336f,          0.879142973793104f);

static const float3x3 kAgXOutputMatrix = float3x3(
     1.19687900512017f,    -0.0980208811401368f, -0.0990297440797205f,
    -0.0528968517574562f,   1.15190312990417f,   -0.0989611768448433f,
    -0.0529716355144438f,  -0.0980434501171241f,  1.15107367264116f);

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

inline float3 AgXSoftGamutClamp(float3 c) {
    c = max(c, 0.0f);
    float peak = max(c.r, max(c.g, c.b));
    if (peak > 1.0f) {
        const float lum = 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
        const float t   = saturate(1.0f - 1.0f / peak);
        c = lerp(c, lum.xxx, t);
        peak = max(c.r, max(c.g, c.b));
        if (peak > 1.0f) c /= peak;
    }
    return c;
}

// Apply the AgX tone curve after exposure and gamut handling.
float3 AgX(float3 color) {

    color = mul(kAgXInputMatrix, color);

    const float minEv = -12.47393f;
    const float maxEv =  4.026069f;
    color = clamp(log2(max(color, 1e-10f)), minEv, maxEv);
    color = (color - minEv) / (maxEv - minEv);

    color = AgXDefaultContrastApprox(color);

    color = mul(kAgXOutputMatrix, color);

    return AgXSoftGamutClamp(color);
}


inline float3 DlssDecode(float3 r, float exposure) {
    return max(r, 0.0f) / max(exposure, 1e-8f);
}

inline float3 ScrubNonFinite(float3 c) {
    return (any(isnan(c)) || any(isinf(c))) ? float3(0, 0, 0) : c;
}

static const float AE_KEY_VALUE = 0.18f;

float ReadExposure() {
    const float smoothedLog2Lum = asfloat(gAutoExpose.Load(AE_OFFS_SMOOTHED));
    return AE_KEY_VALUE / max(exp2(smoothedLog2Lum), 1e-6f);
}

inline uint DitherHash32(uint h) {
    h ^= h >> 16; h *= 0x85EBCA6Bu;
    h ^= h >> 13; h *= 0xC2B2AE35u;
    h ^= h >> 16;
    return h;
}

inline float DitherUniform01(uint h) {
    return float(h & 0x00FFFFFFu) * (1.0f / float(0x01000000u));
}

inline float TriDither(uint seed) {

    return DitherUniform01(DitherHash32(seed))
         - DitherUniform01(DitherHash32(seed + 0x9E3779B9u));
}

inline float3 ApplyOutputDither(float3 c, uint2 pix, uint frameSeed) {
    const uint base = pix.x * 0xCC9E2D51u + pix.y * 0x1B873593u + frameSeed;
    return c + (1.0f / 255.0f) * float3(
        TriDither(base),
        TriDither(base + 0x6F7E1437u),
        TriDither(base + 0xB1C5A8F1u));
}

static const float RCAS_LIMIT = 0.25f - (1.0f / 16.0f);

float3 TonemappedCleanAt(int2 p, float exposure) {
    p = clamp(p, int2(0, 0), int2((int)IMG_W - 1, (int)IMG_H - 1));
    float3 c = DlssDecode(g_dlssOutput[p].xyz, exposure);
    c = ScrubNonFinite(c);
    return AgX(c * exposure);
}

// Sharpen the exposed neighborhood while preserving finite values.
float3 RcasSharpen(uint2 pix, float exposure, float3 e, float sharpness) {

    const float3 b = TonemappedCleanAt(int2(pix) + int2( 0, -1), exposure);
    const float3 d = TonemappedCleanAt(int2(pix) + int2(-1,  0), exposure);
    const float3 f = TonemappedCleanAt(int2(pix) + int2( 1,  0), exposure);
    const float3 h = TonemappedCleanAt(int2(pix) + int2( 0,  1), exposure);

    const float3 mn4 = min(min(b, d), min(f, h));
    const float3 mx4 = max(max(b, d), max(f, h));

    const float3 hitMin = min(mn4, e) / max(4.0f * mx4, 1e-4f);
    const float3 hitMax = (1.0f - max(mx4, e)) / max(4.0f * mn4 - 4.0f, -1e-4f);
    const float3 lobeRGB = max(-hitMin, hitMax);

    float lobe = max(-RCAS_LIMIT,
                     min(max(max(lobeRGB.r, lobeRGB.g), lobeRGB.b), 0.0f)) * sharpness;

    return saturate((lobe * (b + d + f + h) + e) / (4.0f * lobe + 1.0f));
}

float3 DlssInputDebugView(uint2 px, uint layer)
{
    uint rw, rh;
    g_dlssInput.GetDimensions(rw, rh);
    const uint2 rpx = min((px * uint2(rw, rh)) / uint2(IMG_W, IMG_H),
                          uint2(rw - 1u, rh - 1u));

    const float winNear = f16tof32(dbg_dlssDepthWin & 0xFFFFu);
    const float winFar  = f16tof32(dbg_dlssDepthWin >> 16);
    const float winInv  = 1.0f / max(winFar - winNear, 1e-3f);

    float3 v = float3(0, 0, 0);
    switch (layer)
    {
    case 1u:  { const float3 c = max(g_dlssInput[rpx].rgb, 0.0f);         v = c / (1.0f + Luma(c)); } break;
    case 2u:  { const float3 c = max(g_dlssOutput[px].rgb, 0.0f);         v = c / (1.0f + Luma(c)); } break;
    case 3u:  {

                const float d = g_dlssDepth[rpx];
                const float n = DLSS_GUIDE_DEPTH_NEAR, f = DLSS_GUIDE_DEPTH_FAR;
                const float z = (n * f) / max(d * (f - n) + n, 1e-7f);
                v = saturate((z - winNear) * winInv).xxx; } break;
    case 4u:  { const float2 m = g_dlssMVec[rpx];     v = float3(saturate(0.5f + m * 0.1f), 0.5f); } break;
    case 5u:  v = g_dlssNormals[rpx].xyz * 0.5f + 0.5f; break;
    case 6u:  v = g_dlssDiffuseAlbedo[rpx].rgb; break;
    case 7u:  v = g_dlssSpecularAlbedo[rpx].rgb; break;
    case 8u:  v = g_dlssRoughness[rpx].xxx; break;
    case 9u:  { const float2 m = g_dlssSpecMVec[rpx]; v = float3(saturate(0.5f + m * 0.1f), 0.5f); } break;
    case 10u: v = saturate((g_dlssSpecHitDist[rpx] - winNear) * winInv).xxx; break;
    case 11u: v = g_dlssTransparency[rpx].rgb; break;
    case 12u: { const float3 c = max(g_dlssColorPreTrans[rpx].rgb, 0.0f);  v = c / (1.0f + Luma(c)); } break;
    case 13u: v = g_dlssBiasHint[rpx].xxx; break;
    default:  break;
    }

    return sRGBGammaCorrection(saturate(ScrubNonFinite(v)));
}

[numthreads(8, 4, 1)]
// Combine denoised layers, tone mapping, and output dithering.
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= IMG_W || DTid.y >= IMG_H) return;

    const float exposure = ReadExposure();
    if((rs_flags & LT_FLAG_DEBUG)!=0u) {
        uint rw,rh;g_dlssInput.GetDimensions(rw,rh);
        uint2 rpx=min((DTid.xy*uint2(rw,rh))/uint2(IMG_W,IMG_H),uint2(rw-1u,rh-1u));
        uint pixel=MapPixelID(uint2(rw,rh),rpx);float3 color=0.02f;
        if((load_flagsWord(g_sample_current,pixel)&SD_FLAG_NOBOUNCE)==0u) {
            SDRecord surface=load_SD(g_sample_current,pixel);
            color=LTC_DebugColor(surface.x1,surface.n1_s);
        }
        gOutput[uint3(DTid.xy,3)]=float4(color,0);return;
    }
    if (SHARC_DEBUG_MODE != 0u)
    {
        uint rw, rh;
        g_dlssInput.GetDimensions(rw, rh);
        uint2 rpx = min((DTid.xy * uint2(rw, rh)) / uint2(IMG_W, IMG_H),
            uint2(rw - 1u, rh - 1u));
        float4 debugValue = gScratchPing[uint3(rpx, SHARC_DEBUG_SCRATCH)];
        float3 color = debugValue.w > 0.0f ? AgX(ScrubNonFinite(debugValue.rgb) * exposure) : debugValue.rgb;

        gOutput[uint3(DTid.xy, 3)] = float4(color, 0.0f);
        return;
    }

    float3 clean  = DlssDecode(g_dlssOutput[DTid.xy].xyz, exposure);
    float3 refl   = float3(0, 0, 0);
    float3 noisy  = float3(0, 0, 0);
    float3 gt     = float3(0, 0, 0);
    float3 albedo = float3(0, 0, 0);
    if (SHADING_DEBUG_SLICES) {
        noisy  = gScratchPing[uint3(DTid.xy, 1)].rgb;
        gt     = gPermanentData[DTid.xy].rgb;
        albedo = gOutput[uint3(DTid.xy, 5)].xyz;
    }

    noisy = ScrubNonFinite(noisy);
    clean = ScrubNonFinite(clean);
    gt    = ScrubNonFinite(gt);

    noisy = AgX(noisy * exposure);
    clean = AgX(clean * exposure);
    gt    = AgX(gt    * exposure);
    refl  = AgX(refl  * exposure);

    albedo = sRGBGammaCorrection(albedo);

    if (pp_sharpness > 0.0f)
        clean = RcasSharpen(DTid.xy, exposure, clean, pp_sharpness);

    const uint frameSeed = asuint(time);
    noisy  = ApplyOutputDither(noisy,  DTid.xy, frameSeed);
    clean  = ApplyOutputDither(clean,  DTid.xy, frameSeed);
    gt     = ApplyOutputDither(gt,     DTid.xy, frameSeed);
    refl   = ApplyOutputDither(refl,   DTid.xy, frameSeed);
    albedo = ApplyOutputDither(albedo, DTid.xy, frameSeed);

    gOutput[uint3(DTid.xy, 0)] = float4(noisy, 0.0f);

    gOutput[uint3(DTid.xy, 1)] = float4(clean, 1.0f);
    gOutput[uint3(DTid.xy, 2)] = float4(gt, 0.0f);

    const uint dlssLayer = dbg_dlssLayer & DLSS_DBG_LAYER_MASK;
    gOutput[uint3(DTid.xy, 3)] = (dlssLayer != 0u)
        ? float4(DlssInputDebugView(DTid.xy, dlssLayer), 0.0f)
        : float4(0.0f, 0.0f, 0.0f, 0.0f);
    gOutput[uint3(DTid.xy, 4)] = float4(refl, 0.0f);
    gOutput[uint3(DTid.xy, 5)] = float4(albedo, 0.0f);
}
