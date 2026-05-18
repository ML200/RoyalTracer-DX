#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//PRIMARY VOLUMETRIC CLOUD PASS — UNIFIED ATMOSPHERE+CLOUD MARCH
//====================================
//For every primary-miss pixel (g_sample_current holds the sky sentinel
//instID=0xFFFFFFFFu) this pass runs a single ray march that integrates
//BOTH the atmospheric scatter (Rayleigh/Mie/ozone) AND the cloud scatter
//(Watoo phase + Wrenninge MS octaves) against the combined extinction.
//
//Three phases share the running (inscatter, transmittance) state:
//  Phase 1: observer → cloud-shell entry, atmosphere only
//  Phase 2: combined march through cloud shell with hull-skip
//  Phase 3: cloud-shell exit → atmosphere top (or planet hit), atmosphere only
//
//Planet body / stars / nightBase are added on top, attenuated by the
//combined transmittance. Sun disc lives in output_primary (from raygen)
//and gets the same combined-T attenuation through the shading composite.
//
//Writes:
//  gScratchPing[..,10] = full sky pixel value (unified inscatter +
//                       background_behind * combined_T)
//  gScratchPing[..,11] = combined atmosphere+cloud transmittance
//  gScratchPing[..,12] = (cloudHitPos.xyz, cloudHitDistKm) — transmittance-
//                       weighted mean cloud depth for DLSS RR MV/depth
//
//Non-sky pixels write identity (zero radiance, unit transmittance) so
//the shading composite is a coherent no-op there.
//
//Ray generation uses InitCameraRayDoF so bokeh blur applies consistently
//to the unified sky+cloud layer.

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
        gScratchPing[uint3(pixel, 12)] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    //Thin-lens DoF camera ray. rayOrigin is the lens position in shifted
    //world space, rayDir aims at the focal point. For the cloud layer
    //(km away) the sub-meter lens-origin offset is negligible compared to
    //the hit distance, but the DIRECTION variation is what produces bokeh
    //on the cloud — that's why we use the DoF sampler instead of pinhole.
    uint   seed = initRandomData(pixel, uint2(0, 0), (uint)time, 71u);
    float3 rayOrigin;
    float3 rayDir;
    InitCameraRayDoF(pixel, uint2(IMG_W, IMG_H), seed, rayOrigin, rayDir);

    //Atmosphere observer = lens position. Sub-meter offset from the camera
    //center is unmeasurable in atmospheric km units.
    SetSkyObserver(rayOrigin + sceneOriginWorld);

    const SunState S = ComputeSunState();

    //Unified march: atmosphere + cloud integrated against combined sigma_t
    //in one loop. Returns the scattered radiance and the combined
    //transmittance from observer to atmosphere exit (or planet hit).
    float3 combinedTr;
    bool   hitPlanet;
    float  cloudHitDistKm;
    float3 unifiedInscatter = EvaluateAtmosphereAndClouds(
        rayDir, S.dirWS, ATMOS_SOLAR_IRRADIANCE * SKY_INTENSITY,
        combinedTr, hitPlanet, cloudHitDistKm);

    //Background behind the unified march — planet body / stars / airglow,
    //all attenuated by the combined transmittance. The star shield uses
    //the unified inscatter to fade stars under bright atmosphere or cloud.
    float3 backgroundBehind = EvaluateSkyBackgroundBehind(
        rayDir, S, hitPlanet, unifiedInscatter);

    //Full sky pixel value goes into the cloud composite slot. The shading
    //pass then computes output_primary = (sun_disc from raygen) * cloudTr
    //+ cloudL, which works out to sun*combinedTr + (unified + bg*combinedTr).
    float3 skyPixel = unifiedInscatter + backgroundBehind * combinedTr;

    //Reconstruct the cloud hit position for DLSS MV/depth. Zero distance
    //(clear-sky pixels) degenerates to the lens origin; the shading pass
    //branches on cloudHitDistKm > 0 to use rotation-only sky MV instead.
    const float3 cloudHitPos = rayOrigin + rayDir * (cloudHitDistKm * WORLD_UNITS_PER_KM);

    gScratchPing[uint3(pixel, 10)] = float4(skyPixel,    0.0f);
    gScratchPing[uint3(pixel, 11)] = float4(combinedTr,  0.0f);
    gScratchPing[uint3(pixel, 12)] = float4(cloudHitPos, cloudHitDistKm);
}
