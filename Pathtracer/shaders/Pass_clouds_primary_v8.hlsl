#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//PRIMARY VOLUMETRIC CLOUD PASS
//====================================
//Decoupled from raygen. For every primary-miss pixel (where g_sample_current
//holds the sky sentinel instID=0xFFFFFFFFu) this pass reconstructs the
//camera ray via the SAME thin-lens DoF sampler the surface raygen uses, then
//runs the full Skybolt step-and-retract cloud integrator (EvaluateClouds).
//Writes:
//
//  gScratchPing[..,10] = cloudL    (cloud in-scatter radiance)
//  gScratchPing[..,11] = cloudTr   (cloud transmittance, RGB)
//  gScratchPing[..,12] = (worldHitPos.xyz, distKm)
//                       transmittance-weighted cloud hit position in shifted
//                       world space + ray distance in km. Distance is 0 if
//                       the ray accumulated no cloud weight (clear sky).
//                       Pass_shading_v8 reads this to write proper MV/depth
//                       /normal/albedo into the DLSS RR inputs for cloud
//                       pixels — without it, the camera-rotation-only sky MV
//                       smears clouds badly under camera translation.
//
//Non-sky pixels write the identity (cloudL=0, cloudTr=1, hitDist=0) so
//shading can apply the composite unconditionally without a branch — keeping
//the inner loop coherent. Sky pixels detect via load_instID == 0xFFFFFFFFu.
//
//Ray generation uses InitCameraRayDoF so bokeh blur applies consistently to
//the cloud layer the same way it does to surfaces. DLSS RR denoises the
//residual per-step march variance via the MV trick the shading pass uses
//for sky pixels.

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

    float3 cloudTr;
    float  cloudHitDistKm;
    float3 cloudL = EvaluateClouds(rayDir, S.dirWS,
                                   ATMOS_SOLAR_IRRADIANCE * SKY_INTENSITY,
                                   cloudTr, cloudHitDistKm);

    //Reconstruct the cloud hit position in shifted world space. Multiply
    //distance-in-km by WORLD_UNITS_PER_KM so the result lives in the same
    //frame as the camera (built from viewI). When the ray accumulated no
    //cloud weight, cloudHitDistKm == 0 and the position degenerates to
    //the lens origin — the shading pass checks the distance to decide.
    const float3 cloudHitPos = rayOrigin + rayDir * (cloudHitDistKm * WORLD_UNITS_PER_KM);

    gScratchPing[uint3(pixel, 10)] = float4(cloudL,         0.0f);
    gScratchPing[uint3(pixel, 11)] = float4(cloudTr,        0.0f);
    gScratchPing[uint3(pixel, 12)] = float4(cloudHitPos,    cloudHitDistKm);
}
