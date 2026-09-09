#define COMPUTE_PASS
#include "Includes_v8.hlsli"

//====================================
//REVERSIBLE PRE-TONEMAP FOR DLSS RR
//====================================
//DLSS RR doesn't tolerate large brightness differences in input — emitters
//used to be saturate'd to [0,1] which destroyed their HDR. Luminance-based
//Reinhard (c * 1/(1+L)) is reversible AND preserves hue across the
//compression, unlike per-channel Reinhard which shifts saturated colors
//(a bright red emitter would have R compressed much more than G/B, leaking
//pink into the DLSS input). Postprocess applies the matching inverse so
//tonemapping happens once, not twice.
inline float3 DlssReinhard(float3 c) {
    c = max(c, 0.0f);
    const float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    return c / (1.0f + lum);
}

//PT-mode bypass (RS_FLAG_PT_ONLY): the clean PT kernel's 1-spp output is
//heavy-tailed — most pixels near zero, rare pixels carrying 1/p spikes.
//Reinhard bounds every spike to <1 BEFORE DLSS-RR averages, and the inverse
//runs AFTER the average, so the spike-carried energy is destroyed (Jensen:
//invert-of-average < average-of-inverts) and the image goes dark even though
//the estimator is unbiased. Under the PT integrator DLSS therefore gets
//LINEAR radiance; Pass_postprocess (DlssDecode) and Pass_autoexpose_reduce
//skip their inverses in lockstep via the same flag.
//
//One failure mode remains on the unbounded linear path: a JITTERED BOUNDARY
//pixel between radiances of very different magnitude (an emitter face vs its
//housing, grazing specular streaks) flips by the full linear ratio every
//frame — per-pixel convergence never settles a pixel that alternates
//surfaces, and preset F visibly loses temporal stability exactly there (sky
//edges stay fine: modest ratio + hard depth separation). So the linear input
//gets an EXPOSURE-RELATIVE luminance cap: cap_linear = CAP / exposure, i.e. a
//fixed display brightness that tracks auto-exposure across day/night. Below
//the cap the path is exactly linear (no Jensen loss); above it is range AgX
//crushes to white anyway. Deliberately lossy — postprocess applies NO inverse
//for it, and the GT slice reads pre-DLSS radiance so it stays unbiased.
#define DLSS_PT_INPUT_LUMA_CAP 64.0f

//MUST match Pass_postprocess_v8::ReadExposure (same AE state, same key).
inline float ReadExposureForCap() {
    const float AE_KEY_VALUE = 0.18f;
    const float smoothedLog2Lum = asfloat(gAutoExpose.Load(AE_OFFS_SMOOTHED));
    return AE_KEY_VALUE / max(exp2(smoothedLog2Lum), 1e-6f);
}

//Non-finite scrub for the DLSS INPUT — the missing half of postprocess's
//display-side ScrubNonFinite. A rare Inf radiance sample (spike math at
//grazing slivers / emitter edges: 1/pdf, 1/d^2, 1/cos) previously reached the
//encoders, and BOTH manufactured NaN from it (Reinhard: Inf/Inf; the PT cap:
//Inf * (cap/Inf) = Inf * 0). One NaN pixel in g_dlssInput permanently poisons
//RR's accumulator at that spot and spreads through the (transformer) model —
//random onset, never heals until a history reset, invisible in the displayed
//image because postprocess scrubs for DISPLAY only. Mirrors
//Pass_postprocess_v8::ScrubNonFinite.
inline float3 ScrubNonFiniteIn(float3 c) {
    return (any(isnan(c)) || any(isinf(c))) ? float3(0, 0, 0) : c;
}

inline float3 DlssEncode(float3 c) {
    c = ScrubNonFiniteIn(c);
    if (PT_ONLY_MODE) {
        //PRE-EXPOSED linear. DLSS-RR carries an exposure contract
        //(DLSSDOptions::preExposure/exposureScale, kBufferTypeExposure) and
        //with everything left at the 1.0 defaults it assumes DISPLAY-REFERRED
        //input — raw scene radiance at an arbitrary absolute scale sits
        //outside the regime its fp16 history accumulates well in, which shows
        //as "stable at first, jitter creeps in as history builds". Multiply
        //by the AE exposure so mid-grey lands at ~0.18 like any shipping
        //game's feed (linear scale — no Jensen energy loss);
        //Pass_postprocess::DlssDecode divides it back out and
        //Pass_autoexpose_reduce compensates its measurement, so the display
        //and the AE fixed point are unchanged. The luma cap then lives in
        //EXPOSED units — a fixed display brightness by construction.
        c = max(c, 0.0f) * ReadExposureForCap();
        const float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
        return (lum > DLSS_PT_INPUT_LUMA_CAP) ? c * (DLSS_PT_INPUT_LUMA_CAP / lum) : c;
    }
    return DlssReinhard(c);
}

//====================================
//EMITTER SPIKE CLAMP (DLSS RR INPUT)
//====================================
//DLSS RR mishandles a large bright emitter (e.g. a lamp filling a big screen
//region) in an otherwise dark scene: DlssReinhard pushes it right against the
//[0,1] rail, where the postprocess inverse amplifies the denoiser's residual
//error into visible artefacts. When RS_FLAG_CLAMP_EMITTERS is on we luminance-
//clamp emitter radiance to DLSS_EMITTER_CAP BEFORE Reinhard, pulling it off the
//rail (Reinhard(16)=0.94 vs ~0.999) so the inverse stays well-behaved.
//
//This is a per-EMITTER-pixel change at render resolution, baked into a normal
//Reinhard value, so the post-DLSS decode stays uniform (default inverse) and is
//immune to DLSS's sub-pixel jitter / temporal reconstruction. Hue is preserved
//(scale the whole colour by cap/luma). AgX is so compressive at the top that a
//clamped emitter and a true one display almost identically, so visible emitter
//brightness is barely affected — only the rail-adjacent error blows up is gone.
//Tunable: raise the cap for brighter emitters (closer to the rail / more risk),
//lower it for safer denoising.
#define DLSS_EMITTER_CAP 16.0f
//RTXPT kSpecularRoughnessThreshold: above this roughness the virtual-image reflection MV is too
//noisy to help, so the glass/mirror spec channel falls back to the surface MV (see spec-MV below).
#define DLSS_SPEC_ROUGHNESS_THRESHOLD 0.25f
inline float3 ClampEmitterLum(float3 c) {
    const float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    return (lum > DLSS_EMITTER_CAP) ? c * (DLSS_EMITTER_CAP / lum) : c;
}

//====================================
//DLSS RR SEE-THROUGH GUIDE (thin glass transmission)
//====================================
//The path tracer shades the glass as the real primary x1, but for DLSS-RR the see-through
//content (windows into buildings, car interiors) is the DOMINANT signal and lives at the depth
//BEHIND the glass. If RR's primary depth + motion vectors describe the glass surface, the
//see-through reprojects with the wrong depth/motion and parallax-SMEARS. So we feed the BEHIND
//surface to RR's diffuse/depth/normal/primary-MV channel (this function resolves it); the glass
//Fresnel reflection stays on RR's SPECULAR channel (spec albedo + the slot-4 reflection-probe
//spec MV) at the call site. Thin glass only (straight through, no bend) — solid refractive glass
//would need a per-interface refraction re-trace, too expensive for the guide pass. Returns the
//input surface unchanged when the primary isn't thin glass; ray-escapes-to-sky -> far guide.
struct RRGuide { float3 x; float3 n; float3 Kd; float Pr; float Pm; uint instID; };

inline RRGuide ResolveRRGuideThroughGlass(SurfaceVertex sv, uint sInstID, float3 camPos)
{
    RRGuide g;
    g.x = sv.x; g.n = sv.n_s; g.Kd = sv.Kd; g.Pr = sv.Pr; g.Pm = sv.Pm; g.instID = sInstID;

    if (!LoadIsThinGlass(sv.matID))
        return g;

    const float3 vdir = normalize(sv.x - camPos);
    float3 tint = LoadTf(sv.matID);                 // primary pane
    float3 ro   = offset_ray(sv.x, -sv.n_s);        // just past the primary pane (exit side)

    //Re-trace loop: each pass finds the closest hit; if it's thin glass — whether NON-opaque
    //(surfaced as a candidate) OR opaque in the BLAS (auto-committed, e.g. a sphere whose load-time
    //Opacity was 1) — accumulate its Tf and step past it, then re-trace. Stop at the first
    //non-thin-glass surface. Robust to the BLAS opaque/non-opaque classification.
    [loop]
    for (uint pane = 0u; pane < 16u; ++pane)
    {
        RayDesc r;
        r.Origin    = ro;
        r.Direction = vdir;
        r.TMin      = 0.00001f;
        r.TMax      = RAY_TMAX_PLANET;

        RayQuery<RAY_FLAG_NONE> q;
        q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, r);
        [loop]
        for (uint it = 0u; q.Proceed() && it < 64u; ++it)
        {
            if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
            {
                const uint ci = q.CandidateInstanceID();
                const uint cp = FlatPrimID(ci, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
                const uint cm = GetMatIDFast(ci, cp);
                if (LoadIsThinGlass(cm) || LoadKd_w(cm) < 1.0f - EPSILON)
                    q.CommitNonOpaqueTriangleHit();
                else if (AlphaCandidateOccludes(ci, cp, q.CandidateTriangleBarycentrics()))
                    q.CommitNonOpaqueTriangleHit();
            }
        }

        if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        {
            //saw through to the background -> far guide. Mark instID = env-miss sentinel so the
            //call site uses a ROTATION-ONLY sky MV: a world-point reprojection of an infinity
            //point (esp. through the glass instID) adds bogus camera-translation parallax.
            g.x = camPos + vdir * cameraFar; g.n = -vdir;
            g.Kd = float3(1.0f, 1.0f, 1.0f); g.Pr = 1.0f; g.Pm = 0.0f;
            g.instID = 0xFFFFFFFFu;
            return g;
        }

        const uint   hi   = q.CommittedInstanceID();
        const uint   hp   = FlatPrimID(hi, q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex());
        const uint   hm   = GetMatIDFast(hi, hp);
        const float3 hpos = ro + vdir * q.CommittedRayT();

        if (LoadIsThinGlass(hm))
        {
            tint *= LoadTf(hm);
            const float3 geoN = CandidateGeoNormalW(hi, hp);
            ro = offset_ray(hpos, (dot(vdir, geoN) >= 0.0f) ? geoN : -geoN);
            continue;
        }

        //first non-thin-glass surface -> the transmission guide surface
        HitInfo bh = EvalSurfaceState(hi, hp, q.CommittedTriangleBarycentrics(), ro, 0u);
        float3 bKd; float bPr, bPm;
        RefetchMaterial(hm, bh.uv, bKd, bPr, bPm, 0u);
        g.x = bh.hitPos; g.n = bh.hitNormal; g.Kd = bKd * tint; g.Pr = bPr; g.Pm = bPm;
        g.instID = hi;
        return g;
    }
    return g;   // exceeded pane cap -> keep glass (rare)
}

//====================================
//SHADING PASS
//====================================
#include "CumulusGuides_v8.hlsli"

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;

    //Floating origin: viewI returns the *shifted* camera (view matrix
    //built in floating origin space). Hit positions are reconstructed via
    //the shifted instance transforms, so camPosWorld stays in the shifted
    //frame and feeds BuildVertex/specular MV below directly.
    const float3 camPosWorld = mul(viewI, float4(0, 0, 0, 1)).xyz;

    float3 output_primary  = gScratchPing[uint3(DTid.xy, 1)].rgb;
    float3 output_indirect = gScratchPing[uint3(DTid.xy, 2)].rgb;

    //(x1 sharp-reflection contribution removed — dead feature, slot 8 retired.
    // x1 direct sun/env removed too — §6.1 made them reservoir candidates, so
    // scratch slot 3 is retired and no longer read here.)

    //TEMP DEBUG (ATM_DEBUG_RING in Constants_v8): raw path-traced components,
    //captured before the atmosphere composite so mode 2 can false-colour them.
    const float3 dbgRawPrimary  = output_primary;
    const float3 dbgRawIndirect = output_indirect;

    //Atmospheric in-scatter and view transmittance from the primary atmosphere pass.
    //Both radiance slots use the same compositing convention as sky/emitter output.
    const float3 atmosphereL = gScratchPing[uint3(DTid.xy, 10)].rgb;
    const float3 atmosphereTr = gScratchPing[uint3(DTid.xy, 11)].rgb;
    // The participating medium is one contribution, independent of the two surface channels.
    float3 accumulation = (output_primary + output_indirect) * atmosphereTr + atmosphereL;

    //Composited radiance -> the postprocess "noisy" debug slice (slot 1), written
    //pre-debug-override so it stays the true radiance. Debug-only view (gated).
#if SHADING_DEBUG_SLICES
    gScratchPing[uint3(DTid.xy, 1)] = float4(accumulation, 0);
#endif

    //==================== TEMP DEBUG: nadir-ring localisation, mode 2 ===========
    //False-colours the path-traced mesh radiance so we can see which term the
    //ring lives in:  RED = indirect / GI,  GREEN = reflection,  BLUE = direct.
    //Report which channel(s) the ring appears in. Remove once the cause found.
    #if ATM_DEBUG_RING == 2
    {
        float dr = 1.0f - exp(-max(Luma(dbgRawIndirect), 0.0f) * 3.0f);
        float dg = 0.0f; // sharp-reflection term removed
        float db = 1.0f - exp(-max(Luma(dbgRawPrimary), 0.0f) * 3.0f);
        accumulation = float3(dr, dg, db);
    }
    #endif
    //============================================================================

    //"gt" running-average reference (gPermanentData -> postprocess gOutput 2).
    //Debug-only comparison view; gated to drop its full-res FP32 read+write.
#if SHADING_DEBUG_SLICES
    bool cameraChanged = false;
    [unroll]
    for (uint i = 0; i < 4; ++i) {
        if (any(view[i] != prevView[i])) cameraChanged = true;
    }
    static const float MAX_SAMPLES     = 1000.0;

    float4 prev        = gPermanentData[DTid.xy];
    float3 prevAvg     = prev.rgb;
    float  prevSamples = prev.a;

    float3 newAvg;
    float  newSamples;
    if (cameraChanged)
    {
        //camera moved, reset running average
        newAvg     = accumulation;
        newSamples = 1.0h;
    }
    else
    {
        newSamples = min(prevSamples + 1.0h, MAX_SAMPLES);
        float invN  = 1.0h / newSamples;
        newAvg     = mad(accumulation - prevAvg, invN, prevAvg);
    }

    gPermanentData[DTid.xy] = float4(newAvg, newSamples);
#endif

    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, DTid.xy);

    //bias hint inputs set per branch, used after
    uint  biasInstID;
    float2 biasMV = float2(0, 0);
    bool  isEmitterSurface = false;

    bool isEmissiveOrSky = load_isEmitter(g_sample_current, pixelIdx);
    if (isEmissiveOrSky)
    {
        //emitter has valid surface, sky sentinel instID=0xFFFFFFFF does not
        uint emInstID = load_instID(g_sample_current, pixelIdx);
        bool hasPosition = (emInstID != 0xFFFFFFFFu);

        if (hasPosition)
        {
            //emitter surface, depth and MV like regular geometry
            float3 emPos  = load_x1(g_sample_current, pixelIdx);
            g_dlssDepth[DTid.xy] = DLSS_GuideDepthFromWorldPos(emPos);

            float2 curPix = DTid.xy;
            //DoF aware MV, pinhole projection on both ends so the lens offset
            //(curPix is the lens jittered pixel, emPos is the lens jittered hit) cancels terrain
            float2 prevPix = GetLastFramePixelCoordinates_Unclamped(
                emPos, prevView, prevProjection, dims, emInstID);
            float2 curPinholePix = GetCurrentFramePixelCoordinates_Unclamped(
                emPos, view, projection, dims, emInstID);
            bool validPrev = (prevPix.x > -1e8f) && (curPinholePix.x > -1e8f);
            float2 emMV = validPrev ? (prevPix - curPinholePix) : float2(0, 0);
            g_dlssMVec[curPix] = emMV;
            biasMV = emMV;
            isEmitterSurface = true;

            //Use the same stored shading normal as the renderer and ReSTIR.
            const float3 emNormal = load_n1_s_with_instID(g_sample_current, pixelIdx, emInstID);
            //PACKED normal-roughness (ePacked, roughness in .w) — the
            //convention every shipping RR title uses (Cyberpunk, RTXPT);
            //the separate-roughness eUnpacked path is rarely exercised and
            //proved preset-F-sensitive. The standalone g_dlssRoughness
            //texture is still written for the inspector, but RR now reads
            //roughness from this .w. Emitter/sky pixels carry
            //roughness 1 (matches their g_dlssRoughness writes).
            g_dlssNormals[DTid.xy] = float4(emNormal, 1.0f);
        }
        else
        {
            g_dlssDepth[DTid.xy] = 0.0f;
            g_dlssNormals[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 1.0f);   //packed roughness (.w)

            //Sky / planet body MV: rotation only reprojection. Sky is
            //at infinite distance so camera translation between frames
            //must not contribute to its apparent motion — only
            //rotation does. The "camPos + worldDir * cameraFar" trick
            //breaks at orbital flight speeds where camera translation
            //per frame can exceed cameraFar, making the far point
            //arbitrarily close and the MV dominated by translation.
            //Direction-only reprojection sidesteps the issue.
            float2 d = ((float2(DTid.xy) + 0.5f) / dims) * 2.0f - 1.0f;
            float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
            float3 worldDir = normalize(mul(viewI, float4(target.xyz, 0)).xyz);
            float3 prevViewDir = mul(prevView, float4(worldDir, 0)).xyz;
            float2 skyMV = float2(0.0f, 0.0f);
            if (prevViewDir.z < 0.0f) {
                float4 prevClip = mul(prevProjection,
                                      float4(prevViewDir * cameraFar, 1.0f));
                if (prevClip.w > 0.0f) {
                    float2 prevNdc = prevClip.xy / prevClip.w;
                    float2 prevUV  = float2(prevNdc.x * 0.5f + 0.5f,
                                            0.5f - prevNdc.y * 0.5f);
                    float2 prevPix = prevUV * dims - 0.5f;
                    //NO jitter term: motionVectorsJittered=eFalse promises
                    //jitter-free MVs, and the surface path's pinhole-both-
                    //ends MVs are jitter-free (static camera -> exactly 0).
                    //The old "- jitter" put MV = -jitter on every sky pixel
                    //of a static frame — per-frame random ±0.5px motion on
                    //sky while geometry read 0, i.e. a permanent MV
                    //disagreement along every geometry/sky boundary.
                    skyMV = prevPix - float2(DTid.xy);
                }
            }
            g_dlssMVec[DTid.xy] = skyMV;
            biasMV = skyMV;

        }

        biasInstID = emInstID;
        g_dlssSpecularAlbedo[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        //Emitter guide uses its emitted RGB before atmosphere and color encoding,
        //clamped to [0,1]. Sky keeps a neutral diffuse-albedo guide.
        const float3 emitterAlbedo = saturate(ScrubNonFiniteIn(dbgRawPrimary));
        g_dlssDiffuseAlbedo[DTid.xy] = float4(hasPosition ? emitterAlbedo : float3(0.5f, 0.5f, 0.5f), 0.0f);
        g_dlssRoughness[DTid.xy] = 1.0f;
        //Separate fp16 distance cap; the guide camera now spans planetary distances.
        g_dlssSpecHitDist[DTid.xy] = hasPosition ? 0.0f : DLSS_SPEC_HIT_MAX;
        g_dlssSpecMVec[DTid.xy] = float2(0.0f, 0.0f);
        //Reinhard pre-tonemap (reversed in postprocess) so emitters survive AgX.
        //Optionally clamp the emitter spike first so DLSS RR / the inverse don't
        //amplify denoiser error off the [0,1] rail (RS_FLAG_CLAMP_EMITTERS). Gate
        //on hasPosition so only emitter SURFACES (lamps) are clamped, not sky/sun
        //(this branch also handles sky). Drop `&& hasPosition` to include sky.
        float3 emitterRadiance = (CLAMP_EMITTERS_MODE && hasPosition)
                                    ? ClampEmitterLum(accumulation)
                                    : accumulation;
        g_dlssInput[DTid.xy] = float4(DlssEncode(emitterRadiance), 1.0f);
#if SHADING_DEBUG_SLICES
        gOutput[uint3(DTid.xy, 5)] = float4(1.0f, 1.0f, 1.0f, 1.0f);
#endif
    }
    else{
        uint sInstID = load_instID(g_sample_current, pixelIdx);
        float3 sPos  = load_x1(g_sample_current, pixelIdx);
        SurfaceVertex sv = BuildVertex(g_sample_current, pixelIdx, sPos, camPosWorld);

        //DLSS RR channel split for thin glass: the DIFFUSE/depth/primary-MV channel
        //tracks the surface seen THROUGH the glass (transmission, the dominant signal -> no
        //parallax smear), while the SPECULAR channel below stays the GLASS Fresnel reflection.
        //rg == sv for non-glass primaries, so this is a no-op everywhere else.
        const RRGuide rg = ResolveRRGuideThroughGlass(sv, sInstID, camPosWorld);

        //DLSS RR input data (diffuse/geometry channel = the see-through transmission surface)
        g_dlssDepth[DTid.xy] = DLSS_GuideDepthFromWorldPos(rg.x);

        //Specular albedo = the GLASS Fresnel reflection (on sv, NOT the see-through surface).
        //reflW is the reflection throughput used by the spec-MV roughness gate below.
        float3 specularAlbedo = EnvBRDFApprox2(sv.Kd, sv.Pr, sv.Pm, dot(sv.o, sv.n_s));
        float  reflW          = saturate(Luma(specularAlbedo));

        //Use the same stored shading normal as the renderer and ReSTIR.
        g_dlssNormals[DTid.xy] = float4(sv.n_s, sv.Pr);
        //Feed base color directly to the RR diffuse-albedo guide.
        g_dlssDiffuseAlbedo[DTid.xy] = float4(rg.Kd, 1.0f);
        g_dlssRoughness[DTid.xy] = sv.Pr;   // GLASS roughness -> keeps the specular (reflection) channel sharp
        //debug mirror of diffuse albedo passed to DLSS RR
#if SHADING_DEBUG_SLICES
        gOutput[uint3(DTid.xy, 5)] = float4(rg.Kd, 1.0f);
#endif

        float2 curPix = DTid.xy;
        //DoF aware MV, rg.x is the lens jittered hit so its pinhole projection differs
        //from curPix. Use the pinhole projection of rg.x on both ends so the MV describes
        //pure scene motion in pinhole space, the lens offset cancels terrain
        //terrain uses the same instanceProps reprojection as scene meshes (its transform is a
        //translation, so the MV is correct). rg.x is the see-through surface for thin glass,
        //so the transmission reprojects at its own depth/motion -> no parallax smear.
        float2 prevPix       = GetLastFramePixelCoordinates_Unclamped(rg.x, prevView, prevProjection, dims, rg.instID);
        float2 curPinholePix = GetCurrentFramePixelCoordinates_Unclamped(rg.x, view, projection, dims, rg.instID);

        bool validPrev = (prevPix.x > -1e8f) && (curPinholePix.x > -1e8f);

        float2 mvPixels = validPrev ? (prevPix - curPinholePix) : float2(0.0, 0.0);

        //see-through escaped to SKY (rg.instID == env-miss sentinel): rotation-only MV, same as
        //the direct-sky path. Sky is at infinity so camera TRANSLATION must not move it; the
        //world-point reprojection above (a cameraFar point) adds parallax and breaks the MV.
        if (rg.instID == 0xFFFFFFFFu)
        {
            float2 dSky = ((float2(DTid.xy) + 0.5f) / dims) * 2.0f - 1.0f;
            float4 tSky = mul(projectionI, float4(dSky.x, -dSky.y, 1, 1));
            float3 worldDirSky  = normalize(mul(viewI, float4(tSky.xyz, 0)).xyz);
            float3 prevViewDirSky = mul(prevView, float4(worldDirSky, 0)).xyz;
            float2 skyMV = float2(0.0f, 0.0f);
            if (prevViewDirSky.z < 0.0f)
            {
                float4 prevClipSky = mul(prevProjection, float4(prevViewDirSky * cameraFar, 1.0f));
                if (prevClipSky.w > 0.0f)
                {
                    float2 prevNdcSky = prevClipSky.xy / prevClipSky.w;
                    float2 prevUVSky  = float2(prevNdcSky.x * 0.5f + 0.5f, 0.5f - prevNdcSky.y * 0.5f);
                    float2 prevPixSky = prevUVSky * dims - 0.5f;
                    //NO jitter term — same rationale as the pure-sky path
                    //above: MVs are jitter-free by contract.
                    skyMV = prevPixSky - float2(DTid.xy);
                }
            }
            mvPixels = skyMV;
        }

        g_dlssMVec[curPix] = mvPixels;

        //disocclusion keys on the GLASS instID (g_sample_last stores the glass primary).
        biasInstID = sInstID;
        biasMV = mvPixels;

        //specular albedo written to the GLASS-reflection channel (specularAlbedo computed above,
        //on sv — the half that broke when earlier see-through code used the behind surface here).
        g_dlssSpecularAlbedo[DTid.xy] = float4(specularAlbedo, 0.0f);

        //Spec hit distance from the slot-4 reflection PROBE (deterministic
        //mirror ray), NOT the reservoir's reconnection vertex: x2 is the
        //winning GI sample's random bounce — per-pixel NOISE on diffuse
        //surfaces, which destabilizes presets that lean on hit-dist
        //reprojection (preset F). virtualPos is the mirrored reflection
        //hit and mirroring preserves the distance to sv.x, so
        //|virtualPos - x1| IS the reflection hit distance. Probe miss
        //(sky reflection) parks on the guide far plane. Linear metres,
        //clamped for the R16F target (fp16 max 65504). Read hoisted here;
        //the spec-MV block below reuses it.
        const float4 reflData   = gScratchPing[uint3(DTid.xy, 4)];
        const uint   reflInstID = asuint(reflData.w);
        g_dlssSpecHitDist[DTid.xy] = (reflInstID != 0xFFFFFFFFu)
            ? min(length(reflData.xyz - sv.x), DLSS_SPEC_HIT_MAX)
            : DLSS_SPEC_HIT_MAX;

        //Spec-MV fallback = the GLASS SURFACE motion (NOT the see-through MV): a rough or sky
        //reflection is anchored to the glass and must track it. sv.x is the glass primary; off
        //glass rg.x==sv.x so this equals mvPixels (no change for opaque/metal primaries).
        float2 surfaceMV = mvPixels;
        if (LoadIsThinGlass(sv.matID))
        {
            float2 gPrev = GetLastFramePixelCoordinates_Unclamped(sv.x, prevView, prevProjection, dims, sInstID);
            float2 gCur  = GetCurrentFramePixelCoordinates_Unclamped(sv.x, view, projection, dims, sInstID);
            if (gPrev.x > -1e8f && gCur.x > -1e8f)
                surfaceMV = gPrev - gCur;
        }

        //Virtual-reflection MV from the slice-4 probe, used ONLY for SHARP reflections
        //(sv.Pr < kSpecularRoughnessThreshold, per RTXPT). Rough reflections blur toward the
        //surface and a probe-miss (sky) has no virtual hit -> both keep surfaceMV.
        float2 specMV = surfaceMV;
        if (reflW > 0.04f && sv.Pr < DLSS_SPEC_ROUGHNESS_THRESHOLD)
        {
            //reflData/reflInstID hoisted above (spec-hit-dist write).
            //reflInstID is the reflected instance unless the probe missed (0xFFFFFFFF sentinel).
            if (reflInstID != 0xFFFFFFFFu)
            {
                //DoF aware spec MV, pinhole on both ends like the surface MV above
                float2 prevRefl = GetLastFramePixelCoordinates_Unclamped(
                    reflData.xyz, prevView, prevProjection, dims, reflInstID);
                float2 curRefl  = GetCurrentFramePixelCoordinates_Unclamped(
                    reflData.xyz, view, projection, dims, reflInstID);
                if (prevRefl.x > -1e8f && curRefl.x > -1e8f)
                    specMV = prevRefl - curRefl;
            }
        }
        g_dlssSpecMVec[DTid.xy] = specMV;

        //same Reinhard pre-tonemap as the emitter path so DLSS RR sees a
        //uniformly bounded input and the postprocess inversion is consistent
#if ATM_DEBUG_RING == 4
        //TEMP DEBUG: planet shading normal as RGB (n_s*0.5+0.5). A correct
        //sphere is a smooth gradient; a flat patch or hard ring in the
        //normal here = the "fucked normals" — the bug is in EvalSurfaceState.
        g_dlssInput[DTid.xy] = float4(sv.n_s * 0.5f + 0.5f, 1.0f);
#else
        g_dlssInput[DTid.xy] = float4(DlssEncode(accumulation), 1.0f);
#endif

    }

    //====================================
    //DLSS EMITTER BIAS MASK
    //====================================
    //Write every pixel every frame: visible emitter surfaces are 1; all other
    //pixels, including sky and reflected/indirect light, are 0.
    ApplyCumulusGuides(DTid.xy,pixelIdx,camPosWorld);
    float cloudOpacity=gScratchPing[uint3(DTid.xy,CUMULUS_NORMAL_SLOT)].w;
    g_dlssBiasHint[DTid.xy] = isEmitterSurface ? (1.0f-cloudOpacity) : 0.0f;

    //====================================
    //DLSS TRANSPARENCY OVERLAY HOOK
    //====================================
    //placeholder side write, host can wire as a post denoise overlay later
    g_dlssTransparency[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);

    if ((rs_flags & RS_FLAG_GUIDE_OFF_ANY) != 0u)
    {
        if ((rs_flags & RS_FLAG_GUIDE_OFF_DEPTH)   != 0u) g_dlssDepth[DTid.xy] = 0.0f;
        if ((rs_flags & RS_FLAG_GUIDE_OFF_MV)      != 0u) g_dlssMVec[DTid.xy] = float2(0.0f, 0.0f);
        if ((rs_flags & (RS_FLAG_GUIDE_OFF_NORMALS | RS_FLAG_GUIDE_OFF_ROUGH)) != 0u)
        {
            float4 nr = g_dlssNormals[DTid.xy];
            if ((rs_flags & RS_FLAG_GUIDE_OFF_NORMALS) != 0u) nr.xyz = float3(0.0f, 0.0f, 0.0f);
            if ((rs_flags & RS_FLAG_GUIDE_OFF_ROUGH)   != 0u) nr.w   = 1.0f;
            g_dlssNormals[DTid.xy] = nr;
        }
        if ((rs_flags & RS_FLAG_GUIDE_OFF_ALBEDO)  != 0u) g_dlssDiffuseAlbedo[DTid.xy]  = float4(1.0f, 1.0f, 1.0f, 1.0f);
        if ((rs_flags & RS_FLAG_GUIDE_OFF_SPECALB) != 0u) g_dlssSpecularAlbedo[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        if ((rs_flags & RS_FLAG_GUIDE_OFF_SPECMV)  != 0u) g_dlssSpecMVec[DTid.xy] = float2(0.0f, 0.0f);
    }

    //====================================
    //DLSS GUIDE SENTINEL
    //====================================
    //Reads back what THIS thread just wrote to the guide textures (post-fp16
    //quantization — a value that overflowed to +INF in the texel is caught
    //here even though the pre-write register value was finite) and reduces
    //anomaly stats into gAutoExpose bytes 32..63 (SENT_OFFS_*, cleared by
    //Pass_camera, read back by the host, shown in the DLSS Inputs panel).
    //Wave-reduced: the all-clean common case costs one atomic set per wave.
    //Mask bits:
    //  0x001 color non-finite          0x002 depth non-finite / outside [0,1]
    //  0x004 MV non-finite             0x008 normal/roughness non-finite
    //  0x010 packed roughness outside [0,1]
    //  0x020 spec-MV non-finite        0x040 diffuse albedo non-finite
    //  0x080 spec albedo non-finite    0x100 |MV| > 256 px
    //  0x200 |spec MV| > 256 px
    {
        const float4 sCol  = g_dlssInput[DTid.xy];
        const float  sDep  = g_dlssDepth[DTid.xy];
        const float2 sMV   = g_dlssMVec[DTid.xy];
        const float4 sNR   = g_dlssNormals[DTid.xy];
        const float2 sSMV  = g_dlssSpecMVec[DTid.xy];
        const float4 sAlb  = g_dlssDiffuseAlbedo[DTid.xy];
        const float4 sSAlb = g_dlssSpecularAlbedo[DTid.xy];

        const float sLum   = 0.2126f * sCol.x + 0.7152f * sCol.y + 0.0722f * sCol.z;
        const float sMvMag = max(abs(sMV.x),  abs(sMV.y));
        const float sSmMag = max(abs(sSMV.x), abs(sSMV.y));

        uint bad = 0u;
        if (any(isnan(sCol.rgb))  || any(isinf(sCol.rgb)))    bad |= 0x001u;
        if (isnan(sDep) || isinf(sDep) || sDep < 0.0f || sDep > 1.0f)
                                                              bad |= 0x002u;
        if (any(isnan(sMV))       || any(isinf(sMV)))         bad |= 0x004u;
        if (any(isnan(sNR))       || any(isinf(sNR)))         bad |= 0x008u;
        if (sNR.w < 0.0f || sNR.w > 1.0f)                     bad |= 0x010u;
        if (any(isnan(sSMV))      || any(isinf(sSMV)))        bad |= 0x020u;
        if (any(isnan(sAlb.rgb))  || any(isinf(sAlb.rgb)))    bad |= 0x040u;
        if (any(isnan(sSAlb.rgb)) || any(isinf(sSAlb.rgb)))   bad |= 0x080u;
        if (sMvMag > 256.0f)                                  bad |= 0x100u;
        if (sSmMag > 256.0f)                                  bad |= 0x200u;

        //first anomalous pixel of the frame (0 = clean); +1 bias so (0,0) is
        //distinguishable from "none"
        if (bad != 0u)
            gAutoExpose.InterlockedCompareStore(SENT_OFFS_FIRSTBAD, 0u,
                                                ((DTid.y + 1u) << 16) | (DTid.x + 1u));

        const bool  nearCap = PT_ONLY_MODE && (sLum >= DLSS_PT_INPUT_LUMA_CAP * 0.999f);
        const uint  wMask   = WaveActiveBitOr(bad);
        const float wLum    = WaveActiveMax(max(sLum, 0.0f));
        const float wMv     = WaveActiveMax(sMvMag);
        const float wSmv    = WaveActiveMax(sSmMag);
        const uint  wCap    = WaveActiveCountBits(nearCap);
        const uint  wBad    = WaveActiveCountBits(bad != 0u);
        if (WaveIsFirstLane()) {
            if (wMask != 0u) gAutoExpose.InterlockedOr(SENT_OFFS_MASK, wMask);
            gAutoExpose.InterlockedMax(SENT_OFFS_MAXLUMA,   asuint(wLum));
            gAutoExpose.InterlockedMax(SENT_OFFS_MAXMV,     asuint(wMv));
            gAutoExpose.InterlockedMax(SENT_OFFS_MAXSPECMV, asuint(wSmv));
            if (wCap != 0u) gAutoExpose.InterlockedAdd(SENT_OFFS_CAPCOUNT, wCap);
            if (wBad != 0u) gAutoExpose.InterlockedAdd(SENT_OFFS_BADCOUNT, wBad);
        }
    }
}
