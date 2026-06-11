#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//PRIMARY VOLUMETRIC CLOUD PASS — UNIFIED ATMOSPHERE+CLOUD MARCH
//====================================
//Runs for EVERY primary pixel — sky AND mesh hits — and writes the cloud
//composite slots that the shading pass reads. The march integrates BOTH
//the atmospheric scatter (Rayleigh/Mie/ozone) AND the cloud scatter (Watoo
//phase + Wrenninge MS octaves) against the combined extinction:
//
//  Sky pixel (instID == 0xFFFFFFFFu)
//    → march from observer to atmosphere top (or planet body), three phases
//      sharing the running (inscatter, transmittance) state:
//        Phase 1: observer → cloud-shell entry, atmosphere only
//        Phase 2: combined march through cloud shell with hull-skip
//        Phase 3: cloud-shell exit → atmosphere top / planet hit
//      Planet body / stars / nightBase are added behind, attenuated by
//      combined Tr. Sun disc lives in output_primary (from raygen) and
//      picks up the same combined-T attenuation via the shading composite.
//
//  Mesh pixel (instID != sentinel)
//    → march only from observer to the mesh hit distance (load_hitT). No
//      background behind — the mesh occludes the planet/stars/airglow.
//      The shading composite uses slot 10/11 to compose the mesh radiance:
//          finalColor = meshRadiance * combinedTr + unifiedInscatter
//      which is the cloud-occluded equivalent of the old aerial-perspective
//      formula and gives clouds in front of the mesh proper partial /
//      full occlusion.
//
//Writes:
//  gScratchPing[..,10] = cloudL (unified in-scatter, plus background-behind
//                        for sky pixels only)
//  gScratchPing[..,11] = combined atmosphere+cloud transmittance from camera
//                        to the march end (atmosphere top / planet / mesh)
//  gScratchPing[..,12] = (cloudHitPos.xyz, cloudHitDistKm) — DETERMINISTIC
//                        macro-shape Hillaire mean cloud depth. Set on both
//                        sky and mesh pixels when a cloud is in the line
//                        of sight; cloudHitDistKm = 0 signals no cloud.
//  gScratchPing[..,13] = (cloudNormalWS.xyz, cloudAlpha) — outward-facing
//                        normal at the cloud representative depth in xyz;
//                        w channel carries the cloud-only accumulated
//                        opacity (1 - macro-shape transmittance) so the
//                        shading pass can pick cloud-vs-mesh DLSS dominance
//                        from a signal that is independent of atmospheric
//                        in-scatter on long mesh rays.
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

    //Sky vs mesh decides three things:
    //  1. Whether the march terminates at the atmosphere top / planet body
    //     (sky) or at the mesh hit distance from raygen (mesh).
    //  2. Whether the cloud g-buffer (depth + normal in slots 12/13) is
    //     written for DLSS RR (sky pixels with clouds) or left as identity
    //     so the shading pass falls back to the mesh's own depth/normal.
    //  3. Whether the planet body / stars / airglow are added behind the
    //     atmosphere (sky pixel sees them through the gaps; mesh occludes
    //     them entirely so backgroundBehind drops terrain).
    const uint instID    = load_instID(g_sample_current, pixelIdx);
    const bool isMeshHit = (instID != 0xFFFFFFFFu);

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

    //Underground camera (below the analytic planet surface): sky, clouds,
    //sun disc and aerial perspective are all geometrically invisible from
    //inside the planet body — write the identity composite and skip the
    //march so underground scenes aren't lit by a sky that isn't there.
    //Mesh pixels: combinedTr = 1 passes the scene radiance through with
    //zero added in-scatter. Sky pixels: combinedTr = 0 also extinguishes
    //raygen's sun disc in the shading composite. Slots 12/13 say "no
    //cloud" so DLSS falls back to mesh depth / rotation-only sky MVs.
    if (SkyObserverIsUnderground())
    {
        const float3 trUnder = isMeshHit ? float3(1.0f, 1.0f, 1.0f)
                                         : float3(0.0f, 0.0f, 0.0f);
        gScratchPing[uint3(pixel, 10)] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        gScratchPing[uint3(pixel, 11)] = float4(trUnder,         0.0f);
        gScratchPing[uint3(pixel, 12)] = float4(rayOrigin,       0.0f);
        gScratchPing[uint3(pixel, 13)] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        return;
    }

    const SunState S = ComputeSunState();

    //Mesh distance clip = distance from this pass's lens-ray origin to
    //raygen's exact stored hit position x1. Sub-meter divergence vs raygen's
    //own lens-jittered ray is negligible against km-scale cloud distances.
    //maxMarchKm = 0 means "sky pixel, no mesh clip" inside
    //EvaluateAtmosphereAndClouds.
    const float maxMarchKm = isMeshHit
        ? (length(load_x1(g_sample_current, pixelIdx) - rayOrigin) * (1.0f / WORLD_UNITS_PER_KM))
        : 0.0f;

    //Unified march: atmosphere + cloud integrated against combined sigma_t
    //in one loop. Returns the scattered radiance and the combined
    //transmittance from observer to atmosphere exit, planet hit, or mesh.
    //Radiance only — the DLSS g-buffer (depth + normal) is produced
    //separately below by EvaluateCloudGBuffer so it stays frame-stable.
    //maxCloudHull = max raw cloud hull the march saw; zero gates the
    //g-buffer march below.
    float3 combinedTr;
    bool   hitPlanet;
    float  maxCloudHull;
    float3 unifiedInscatter = EvaluateAtmosphereAndClouds(
        rayDir, S.dirWS, ATMOS_SOLAR_IRRADIANCE * SKY_INTENSITY,
        maxMarchKm,
        combinedTr, hitPlanet, maxCloudHull);

    //Deterministic stable g-buffer (depth + normal) for DLSS RR. Runs for
    //both sky AND mesh pixels — a thick cloud sitting between the camera and
    //a mesh below (the "flying above clouds, ground underneath" case) is the
    //visible surface for that pixel, so DLSS should see the cloud's depth/
    //normal and use the cloud's world-space MV, not the mesh's. The shading
    //pass decides cloud-vs-mesh via the same opacity threshold the sky path
    //uses (cloudOpacity > 0.05 with cloudHitDistKm > 0).
    //
    //The stable march uses macro-shape density only at fixed stratified
    //positions, so the result is the same every frame for the same world
    //geometry — DLSS RR no longer reads cloud-depth jitter as motion.
    //Returns (0, zero) for clear-sky and for cloud-free mesh pixels; the
    //shading pass branches on cloudHitDistKm > 0.
    //
    //For mesh pixels we pass maxMarchKm so the g-buffer march also clips at
    //the mesh distance (mesh inside cloud shell case); otherwise the mean
    //depth could fall past the mesh, where the cloud is no longer visible.
    //
    //cloudAlpha is the cloud-only accumulated opacity (atmosphere extinction
    //excluded). The shading pass uses it instead of luma(cloudL) ratios to
    //decide cloud-vs-mesh DLSS dominance, since cloudL bundles atmospheric
    //in-scatter and would otherwise let bright sky haze register as "cloud"
    //on long mesh rays.
    //Gated on the unified march having seen ANY nonzero cloud hull along
    //this exact ray: the g-buffer march walks the same shell with the same
    //direction-only hull terms, so an empty hull there is empty here too —
    //skipping it saves the full 32-step CloudShapeDensity march on clear-sky
    //and cloud-free mesh pixels. (The radiance march can stride over a small
    //cloud the fixed-step g-buffer march would catch, but in that frame the
    //radiance has no cloud in it either, so depth/alpha = "no cloud" is the
    //CONSISTENT answer for DLSS.)
    float  cloudHitDistKm = 0.0f;
    float3 cloudNormalWS  = float3(0.0f, 0.0f, 0.0f);
    float  cloudAlpha     = 0.0f;
    if (maxCloudHull > 0.0f)
        EvaluateCloudGBuffer(rayDir, maxMarchKm, cloudHitDistKm, cloudNormalWS, cloudAlpha);

    //Background behind the unified march — planet body / stars / airglow,
    //all attenuated by the combined transmittance. Sky pixels see them
    //through the partial cloud cover; mesh pixels occlude them entirely
    //(the mesh is in front of the background sphere by construction).
    //Planet-clipped rays composite against black on purpose — see the
    //"NO analytic ground backstop" note in Clouds_v8.hlsli.
    float3 backgroundBehind = float3(0, 0, 0);
    if (!isMeshHit)
    {
        backgroundBehind = EvaluateSkyBackgroundBehind(
            rayDir, S, hitPlanet, unifiedInscatter);
    }

    //Slot 10 holds the radiance that the shading pass adds to the scene
    //radiance after attenuation by slot 11's transmittance:
    //  finalColor = sceneRadiance * cloudTr + cloudL
    //For sky pixels cloudL bundles unified atmosphere/cloud in-scatter +
    //planet/stars/airglow behind. For mesh pixels it's just the unified
    //in-scatter between camera and mesh (no background behind to add).
    float3 cloudL = isMeshHit
        ? unifiedInscatter
        : (unifiedInscatter + backgroundBehind * combinedTr);

    //Reconstruct the cloud hit position for DLSS MV/depth. Zero distance
    //(clear-sky and mesh pixels) degenerates to the lens origin; the shading
    //pass branches on cloudHitDistKm > 0 to pick the cloud-aware vs
    //rotation-only sky vs mesh MV path.
    const float3 cloudHitPos = rayOrigin + rayDir * (cloudHitDistKm * WORLD_UNITS_PER_KM);

    //==================== TEMP DEBUG: nadir-ring localisation ====================
    //ATM_DEBUG_RING lives in Constants_v8.hlsli. Mode 1 = this block: a false
    //colour view of the unified atmosphere march output (bypasses the mesh):
    //   RED = combinedTr,  GREEN = unifiedInscatter (tone compressed).
    //Mode 1 already showed the ring is NOT here. Mode 2 lives in Pass_shading.
    #if ATM_DEBUG_RING == 1
    {
        float dbgTr  = saturate(dot(combinedTr, float3(0.33333f, 0.33333f, 0.33334f)));
        float dbgIns = dot(unifiedInscatter, float3(0.33333f, 0.33333f, 0.33334f));
        dbgIns       = saturate(1.0f - exp(-max(dbgIns, 0.0f) * 4.0f));
        //*0.5 cancels the slot1+slot2 double-add in the shading composite
        cloudL     = float3(dbgTr, dbgIns, 0.0f) * 0.5f;
        combinedTr = float3(0.0f, 0.0f, 0.0f);
    }
    #endif
    //=============================================================================

    gScratchPing[uint3(pixel, 10)] = float4(cloudL,        0.0f);
    gScratchPing[uint3(pixel, 11)] = float4(combinedTr,    0.0f);
    gScratchPing[uint3(pixel, 12)] = float4(cloudHitPos,   cloudHitDistKm);
    //slot 13.w = cloud-only opacity for the shading pass's mesh DLSS branch.
    gScratchPing[uint3(pixel, 13)] = float4(cloudNormalWS, cloudAlpha);
}
