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

//Soft display-gamut handling for AgX output. Pure saturate per-channel hue-
//shifts saturated emitters (a (1.2, 0.4, 0.2) pixel goes (1, 0.4, 0.2) and
//looks more orange than the original red). We instead desaturate toward
//luminance as the brightest channel overshoots 1.0, then guarantee peak ≤ 1.
//Both the chroma reduction and the final normalize are gentle; pixels already
//in-gamut pass through unchanged.
inline float3 AgXSoftGamutClamp(float3 c) {
    c = max(c, 0.0f);
    float peak = max(c.r, max(c.g, c.b));
    if (peak > 1.0f) {
        const float lum = 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
        const float t   = saturate(1.0f - 1.0f / peak);  // 0 at peak=1, →1 at peak→∞
        c = lerp(c, lum.xxx, t);
        peak = max(c.r, max(c.g, c.b));
        if (peak > 1.0f) c /= peak;
    }
    return c;
}

float3 AgX(float3 color) {
    //sRGB -> AgX working primaries
    color = mul(kAgXInputMatrix, color);

    //log2 encode, clamp the exposure range, normalize to [0,1].
    //Returned to the canonical 4.026 maxEv; widening it (previously 6.0) was a
    //workaround for bright emitters not getting brought into AgX's natural
    //range. Now that auto-exposure measures real HDR (not the Reinhard-
    //bounded image), pre-AgX exposure scales emitters down on its own.
    const float minEv = -12.47393f;
    const float maxEv =  4.026069f;
    color = clamp(log2(max(color, 1e-10f)), minEv, maxEv);
    color = (color - minEv) / (maxEv - minEv);

    //display sigmoid (this IS the EOTF for AgX)
    color = AgXDefaultContrastApprox(color);

    //AgX -> sRGB display primaries, then hue-preserving gamut clamp
    color = mul(kAgXOutputMatrix, color);

    return AgXSoftGamutClamp(color);
}

//====================================
//INVERSE REINHARD (RECOVER HDR AFTER DLSS RR)
//====================================
//Pass_shading_v8.hlsl applies c / (1 + L) (luminance Reinhard) to its DLSS RR
//input so DLSS sees a bounded range (the network mishandles very bright
//emitters otherwise). Postprocess inverts that here so AgX still operates on
//real HDR — without this, emitters get clipped at the shading stage AND
//compressed again by AgX, ending up dimmer than their surroundings.
//Per-channel saturate bounds the recovered HDR at ~10000 nits-equivalent,
//well below AgX's clipping point.
inline float3 InverseDlssReinhard(float3 r) {
    r = saturate(r);
    const float lumR = 0.2126f * r.x + 0.7152f * r.y + 0.0722f * r.z;
    return r / max(1.0f - lumR, 1e-4f);
}

//====================================
//NON-FINITE SCRUB (safety backstop)
//====================================
//A poisoned reservoir sample (a divide that slipped an upstream guard) must not
//reach the display. AgX clamps +inf (log2(max(.,1e-10)) -> clamped to maxEv) but
//PROPAGATES NaN through the sigmoid, so a single NaN pixel renders as garbage.
//This drops non-finite pixels to black. It deliberately does NOT clamp finite
//fireflies -- those are root-caused in the reuse passes (bounded Jacobian), not
//masked here.
inline float3 ScrubNonFinite(float3 c) {
    return (any(isnan(c)) || any(isinf(c))) ? float3(0, 0, 0) : c;
}

//====================================
//EXPOSURE FROM AUTO-EXPOSE STATE
//====================================
//exposure = key / 2^smoothed_log2_lum. AE finalize clamps the log2-lum range,
//so this can't blow up. 0.18 is photographic mid-grey; with AgX maxEv at the
//default 4.026 this places the mean scene luminance at a balanced mid-tone.
static const float AE_KEY_VALUE = 0.18f;

float ReadExposure() {
    const float smoothedLog2Lum = asfloat(gAutoExpose.Load(AE_OFFS_SMOOTHED));
    return AE_KEY_VALUE / max(exp2(smoothedLog2Lum), 1e-6f);
}

//====================================
//OUTPUT DITHER (HIDES 8-BIT BANDING)
//====================================
//Triangular-PDF noise applied in sRGB-encoded space (where the UNORM
//quantization happens). Per-channel independent dither hides banding in
//smooth gradients (twilight sky, dark surfaces) without visible noise on
//mid-tones. Animated by CameraParams.time so it's temporally invisible at
//normal framerates and DLSS-RR accumulates it away.
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
    //triangular PDF in [-1, +1] = U1 - U2 with U1, U2 independent uniform
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

//====================================
//POST-PROCESS PASS
//====================================
[numthreads(8, 4, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;

    const float exposure = ReadExposure();

    //HDR debug slices read from FP32 scratchPing / gPermanentData. Writing raw HDR
    //into the R8G8B8A8 UNORM gOutput would clip to 1.0 and quantize to 8 bits linear
    //before tonemap could ever see it, which is why noisy/gt/nrc/refl used to look
    //unexposed and banded relative to the DLSS clean slice
    float3 noisy  = gScratchPing[uint3(DTid.xy, 1)].rgb +
                    gScratchPing[uint3(DTid.xy, 2)].rgb +
                    gScratchPing[uint3(DTid.xy, 3)].rgb;
    //clean came through DLSS RR which received Reinhard-tonemapped input —
    //undo that here so AgX sees the original HDR
    float3 clean  = InverseDlssReinhard(g_dlssOutput[DTid.xy].xyz);
    float3 gt     = gPermanentData[DTid.xy].rgb;
    float3 nrc    = gScratchPing[uint3(DTid.xy, 9)].rgb;
    float3 refl   = float3(0, 0, 0); // sharp-reflection slice retired (kept black)
    float3 albedo = gOutput[uint3(DTid.xy, 5)].xyz;

    //Scrub non-finite HDR before tonemap so a NaN/inf can't reach the display.
    noisy = ScrubNonFinite(noisy);
    clean = ScrubNonFinite(clean);
    gt    = ScrubNonFinite(gt);
    nrc   = ScrubNonFinite(nrc);

    //AgX filmic tonemap (Troy Sobotka), operating on real HDR (post-exposure).
    //Its sigmoid + output matrix already bake in the sRGB display EOTF, so the
    //result goes straight to the UNORM target with NO sRGBGammaCorrection after.
    //(Temporarily swapped for a plain sRGB encode 2026-06-20 for the SPMIS noise
    //comparison; re-enabled here.)
    noisy = AgX(noisy * exposure);
    clean = AgX(clean * exposure);
    gt    = AgX(gt    * exposure);
    nrc   = AgX(nrc   * exposure);
    refl  = AgX(refl  * exposure);
    //albedo is reflectance in [0,1] - no HDR, no exposure, no tonemap; sRGB only
    albedo = sRGBGammaCorrection(albedo);

    //triangular dither in display space hides 8-bit banding. Animated per
    //frame so DLSS RR accumulates the noise away on the live slice.
    const uint frameSeed = asuint(time);
    noisy  = ApplyOutputDither(noisy,  DTid.xy, frameSeed);
    clean  = ApplyOutputDither(clean,  DTid.xy, frameSeed);
    gt     = ApplyOutputDither(gt,     DTid.xy, frameSeed);
    nrc    = ApplyOutputDither(nrc,    DTid.xy, frameSeed);
    refl   = ApplyOutputDither(refl,   DTid.xy, frameSeed);
    albedo = ApplyOutputDither(albedo, DTid.xy, frameSeed);

    gOutput[uint3(DTid.xy, 0)] = float4(noisy, 0.0f);
    gOutput[uint3(DTid.xy, 1)] = float4(clean, 0.0f);
    gOutput[uint3(DTid.xy, 2)] = float4(gt, 0.0f);
    gOutput[uint3(DTid.xy, 3)] = float4(nrc, 0.0f);
    gOutput[uint3(DTid.xy, 4)] = float4(refl, 0.0f);
    gOutput[uint3(DTid.xy, 5)] = float4(albedo, 0.0f);
}
