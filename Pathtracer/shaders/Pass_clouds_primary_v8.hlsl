#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//PRIMARY VOLUMETRIC CLOUD PASS
//====================================
//Decoupled from raygen. For every primary-miss pixel (where g_sample_current
//holds the sky sentinel instID=0xFFFFFFFFu) this pass reconstructs the
//pinhole camera ray and runs the FULL Skybolt step-and-retract cloud
//integrator (EvaluateClouds). Writes:
//
//  gScratchPing[..,10] = cloudL  (cloud in scatter radiance)
//  gScratchPing[..,11] = cloudTr (cloud transmittance, RGB)
//
//Pass_shading_v8 composites these into the sky/sun background sitting in
//scratch slots 1 & 2 via the standard alpha blend:
//  composited = background * cloudTr + cloudL
//
//Non-sky pixels write the identity (cloudL=0, cloudTr=1) so shading can
//apply the composite unconditionally without a branch — keeping the inner
//loop coherent. Sky pixels detect via load_instID == 0xFFFFFFFFu.
//
//Ray reconstruction uses the un-jittered pinhole projection (no DoF), so
//the cloud pass output is screen-space stable. DLSS RR denoises the
//remaining per-step march variance — the bounce path doesn't matter here
//because primary rays bypass the bounce loop on a miss.

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;
    gDispatchIdx = DTid;

    const uint2  pixel    = DTid.xy;
    const float2 dims     = float2(IMG_W, IMG_H);
    const uint   pixelIdx = MapPixelID(dims, pixel);

    //Default identity so shading composite is a no-op on non-sky pixels.
    //Written unconditionally below in the early-out path.
    const uint instID = load_instID(g_sample_current, pixelIdx);
    if (instID != 0xFFFFFFFFu)
    {
        gScratchPing[uint3(pixel, 10)] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        gScratchPing[uint3(pixel, 11)] = float4(1.0f, 1.0f, 1.0f, 0.0f);
        return;
    }

    //Atmosphere observer state needs the absolute (planet centred) camera
    //position. View matrix is built in shifted (floating origin) space.
    const float3 camPosShift = mul(viewI, float4(0, 0, 0, 1)).xyz;
    SetSkyObserver(camPosShift + sceneOriginWorld);

    //Pinhole reconstruction (no jitter, no DoF). Cloud noise is denoised
    //by DLSS RR via the same MV trick that shading uses for sky pixels,
    //so a stable pinhole direction is exactly what we want here.
    float2 d = ((float2(pixel) + 0.5f) / dims) * 2.0f - 1.0f;
    float4 viewH    = mul(projectionI, float4(d.x, -d.y, 1, 1));
    float3 worldDir = SafeNormalize(mul(viewI, float4(viewH.xyz, 0)).xyz);

    const SunState S = ComputeSunState();

    float3 cloudTr;
    float3 cloudL = EvaluateClouds(worldDir, S.dirWS,
                                   ATMOS_SOLAR_IRRADIANCE * SKY_INTENSITY,
                                   cloudTr);

    gScratchPing[uint3(pixel, 10)] = float4(cloudL,  0.0f);
    gScratchPing[uint3(pixel, 11)] = float4(cloudTr, 0.0f);
}
