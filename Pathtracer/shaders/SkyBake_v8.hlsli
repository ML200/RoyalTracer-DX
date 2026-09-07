#ifndef SKYBAKE_V8_HLSLI
#define SKYBAKE_V8_HLSLI
#include "SkyBakeLayout.h"

//====================================
//PER-FRAME SKY BAKE (regular path tracer)
//====================================
// Pass_pt_skybake_v8.hlsl writes, once per frame, what the bounce and training
// kernels used to compute per call: the sun state of the camera observer
// (ComputeSunStateInline: GetSunDirAndElev plus a 12-step transmittance
// march, run at every sun NEE sample and up to four more times per sky miss)
// and the atmosphere along every direction from that observer
// (IntegrateScattering, a 12-step in-scatter march per sky miss). Pass_pt sets
// SKYBAKE_CONSUMER and reads them; ComputeSunState and SkyAtmosphere
// (SunSampler_v8.hlsli) switch on it. Every other pass keeps its inline
// evaluation, and a pass that moves the observer per vertex (Pass_raygen)
// must never consume the bake.
//
// Storage rides the SPMIS hash-grid buffer: under RS_FLAG_PT_ONLY the host
// clears the SPMIS bits and skips every SPMIS pass, so nothing else touches
// the buffer while the path tracer owns the frame (the lite reuse tables ride
// the SHaRC allocation the same way). Layout in SkyBakeLayout.h:
//   [SKYBAKE_SUN_OFFSET]  SunState, 64 bytes
//   [SKYBAKE_LUT_OFFSET]  sky-view LUT, SKYBAKE_LUT_W x SKYBAKE_LUT_H texels of
//                         {scatter.rgb, planet hit, viewTr.rgb, 0} as 8 halfs
// scatter is the in-scatter in solar-irradiance units (before SKY_INTENSITY),
// viewTr the transmittance along the marched path, planet hit 1 where the
// ray ends on the planet.
//
// Texel mapping: azimuth atan2(z, x) across u, wrapping; elevation across v
// through the square-root mapping (Hillaire 2020), so the rows are densest
// at the horizon, where the in-scatter changes fastest and the planet-hit
// edge lies. At 128 rows the first row above the horizon is 0.007 degrees
// tall; the bilinear blend across a ground-level observer's horizon stays
// far below anything a bounce ray resolves.

float2 SkyBakeUvFromDir(float3 v)
{
    const float az = atan2(v.z, v.x) * (0.5f / PI) + 0.5f;
    const float el = asin(clamp(v.y, -1.0f, 1.0f));
    const float x  = sqrt(abs(el) * (2.0f / PI)) * (el < 0.0f ? -1.0f : 1.0f);
    return float2(az, 0.5f + 0.5f * x);
}

float3 SkyBakeDirFromUv(float2 uv)
{
    const float az = (uv.x - 0.5f) * (2.0f * PI);
    const float x  = uv.y * 2.0f - 1.0f;
    const float el = x * x * (0.5f * PI) * (x < 0.0f ? -1.0f : 1.0f);
    const float ce = cos(el);
    return float3(ce * cos(az), sin(el), ce * sin(az));
}

uint SkyBakeTexelAddress(uint2 texel)
{
    return SKYBAKE_LUT_OFFSET + (texel.y * SKYBAKE_LUT_W + texel.x) * SKYBAKE_LUT_TEXEL_BYTES;
}

void SkyBakeStoreView(uint2 texel, float3 scatter, float3 viewTr, float hitPlanet)
{
    const uint4 w = uint4(
        f32tof16(scatter.x) | (f32tof16(scatter.y) << 16u),
        f32tof16(scatter.z) | (f32tof16(hitPlanet) << 16u),
        f32tof16(viewTr.x)  | (f32tof16(viewTr.y)  << 16u),
        f32tof16(viewTr.z));
    g_spmisBuffer.Store4(SkyBakeTexelAddress(texel), w);
}

void SkyBakeDecodeView(uint4 w, out float3 scatter, out float3 viewTr, out float hitPlanet)
{
    scatter   = float3(f16tof32(w.x & 0xffffu), f16tof32(w.x >> 16u), f16tof32(w.y & 0xffffu));
    hitPlanet = f16tof32(w.y >> 16u);
    viewTr    = float3(f16tof32(w.z & 0xffffu), f16tof32(w.z >> 16u), f16tof32(w.w & 0xffffu));
}

// Bilinear reconstruction: wrapping in azimuth, clamped in elevation.
void SkyBakeLoadView(float3 v, out float3 scatter, out float3 viewTr, out float hitPlanet)
{
    const float2 uv  = SkyBakeUvFromDir(v);
    const float  fx  = uv.x * (float)SKYBAKE_LUT_W - 0.5f;
    const float  fy  = clamp(uv.y * (float)SKYBAKE_LUT_H - 0.5f, 0.0f, (float)(SKYBAKE_LUT_H - 1u));
    const float  x0f = floor(fx);
    const float  wx  = fx - x0f;
    const float  wy  = frac(fy);
    const uint   x0  = (uint)((int)x0f + (int)SKYBAKE_LUT_W) % SKYBAKE_LUT_W;
    const uint   x1  = (x0 + 1u) % SKYBAKE_LUT_W;
    const uint   y0  = (uint)fy;
    const uint   y1  = min(y0 + 1u, SKYBAKE_LUT_H - 1u);
    float3 s00, s10, s01, s11, t00, t10, t01, t11;
    float  h00, h10, h01, h11;
    SkyBakeDecodeView(g_spmisBuffer.Load4(SkyBakeTexelAddress(uint2(x0, y0))), s00, t00, h00);
    SkyBakeDecodeView(g_spmisBuffer.Load4(SkyBakeTexelAddress(uint2(x1, y0))), s10, t10, h10);
    SkyBakeDecodeView(g_spmisBuffer.Load4(SkyBakeTexelAddress(uint2(x0, y1))), s01, t01, h01);
    SkyBakeDecodeView(g_spmisBuffer.Load4(SkyBakeTexelAddress(uint2(x1, y1))), s11, t11, h11);
    scatter   = lerp(lerp(s00, s10, wx), lerp(s01, s11, wx), wy);
    viewTr    = lerp(lerp(t00, t10, wx), lerp(t01, t11, wx), wy);
    hitPlanet = lerp(lerp(h00, h10, wx), lerp(h01, h11, wx), wy);
}

void SkyBakeStoreSunState(SunState S)
{
    g_spmisBuffer.Store4(SKYBAKE_SUN_OFFSET,       uint4(asuint(S.dirWS), asuint(S.elevRad)));
    g_spmisBuffer.Store4(SKYBAKE_SUN_OFFSET + 16u, uint4(asuint(S.cosThetaMax), asuint(S.omega),
                                                         asuint(S.pdf), asuint(S.visible)));
    g_spmisBuffer.Store4(SKYBAKE_SUN_OFFSET + 32u, uint4(asuint(S.radiance), 0u));
    g_spmisBuffer.Store4(SKYBAKE_SUN_OFFSET + 48u, uint4(asuint(S.tint), 0u));
}

SunState SkyBakeLoadSunState()
{
    const uint4 a = g_spmisBuffer.Load4(SKYBAKE_SUN_OFFSET);
    const uint4 b = g_spmisBuffer.Load4(SKYBAKE_SUN_OFFSET + 16u);
    const uint4 c = g_spmisBuffer.Load4(SKYBAKE_SUN_OFFSET + 32u);
    const uint4 d = g_spmisBuffer.Load4(SKYBAKE_SUN_OFFSET + 48u);
    SunState S;
    S.dirWS       = asfloat(a.xyz);
    S.elevRad     = asfloat(a.w);
    S.cosThetaMax = asfloat(b.x);
    S.omega       = asfloat(b.y);
    S.pdf         = asfloat(b.z);
    S.visible     = asfloat(b.w);
    S.radiance    = asfloat(c.xyz);
    S.tint        = asfloat(d.xyz);
    return S;
}

#endif
