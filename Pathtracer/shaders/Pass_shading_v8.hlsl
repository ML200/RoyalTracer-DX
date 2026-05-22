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

//====================================
//SHADING PASS
//====================================
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
    float3 sunDirect       = gScratchPing[uint3(DTid.xy, 3)].rgb;

    //====================================
    //X1 SHARP REFLECTION CONTRIBUTION
    //====================================
    //slot 8 is the durable handoff, debug passes scribble g_NrcInferenceOut later.
    //Read here (above the cloud composite) so the mesh path can attenuate it
    //before postprocess sees it.
    float3 reflContrib = float3(0, 0, 0);
    float  reflAlpha   = 0.0f;
    {
        const float4 reflPack = gScratchPing[uint3(DTid.xy, 8)];
        reflAlpha = reflPack.w;
        if (reflPack.w > 0.0f)
            reflContrib = max(float3(0, 0, 0), reflPack.rgb);
    }

    //TEMP DEBUG (ATM_DEBUG_RING in Constants_v8): raw path-traced components,
    //captured before the cloud composite so mode 2 can false-colour them.
    const float3 dbgRawPrimary  = output_primary;
    const float3 dbgRawIndirect = output_indirect;
    const float3 dbgRawSun      = sunDirect;
    const float3 dbgRawRefl     = reflContrib;

    //====================================
    //CLOUD + ATMOSPHERE COMPOSITE
    //====================================
    //Pass_clouds_primary_v8 wrote cloudL into slot 10 and combined atmosphere
    //+ cloud transmittance into slot 11 for EVERY primary pixel. Sky pixels:
    //cloudL bundles the unified in-scatter plus planet/stars/airglow behind.
    //Mesh pixels: cloudL is the unified in-scatter from camera to mesh hit
    //(no background behind — the mesh occludes it), cloudTr is the camera→
    //mesh atm+cloud transmittance. For mesh pixels this composite replaces
    //the old ApplyAerialPerspective call — the unified march already handled
    //atmospheric scatter/extinction so doing aerial perspective here too
    //would double-count atmosphere and ignore cloud occlusion entirely.
    //
    //Per-slot composite uniform across sky and mesh — slot 1 AND slot 2 both
    //pick up cloudL on top of their attenuated radiance. The doubling is
    //intentional: raygen writes slot1 = slot2 = sun-disc/emission for sky
    //and emitter pixels, so the postprocess sum (slot1+slot2+slot3+refl)
    //naturally counts those contributions twice, which means cloudL must
    //also land twice for a cloud-occluded mesh to read at the SAME apparent
    //brightness as the same cloud seen as a sky pixel. Injecting cloudL
    //only once gave mesh-through-cloud half the cloud brightness, reading
    //as "dimmed mesh" instead of "covered by a bright cloud".
    //
    //Identity case (no cloud, no atmosphere): cloudL = 0 and cloudTr =
    //(1,1,1) so the composite is a no-op.
    {
        const float3 cloudL  = gScratchPing[uint3(DTid.xy, 10)].rgb;
        const float3 cloudTr = gScratchPing[uint3(DTid.xy, 11)].rgb;

        output_primary  = output_primary  * cloudTr + cloudL;
        output_indirect = output_indirect * cloudTr + cloudL;
        sunDirect      *= cloudTr;
        reflContrib    *= cloudTr;

        gScratchPing[uint3(DTid.xy, 1)] = float4(output_primary,  0);
        gScratchPing[uint3(DTid.xy, 2)] = float4(output_indirect, 0);
        gScratchPing[uint3(DTid.xy, 3)] = float4(sunDirect,       0);
        //Preserve reflPack.w (validity flag) so the postprocess gate
        //(reflPack.w > 0.0f ? rgb : 0) still reads the right state.
        gScratchPing[uint3(DTid.xy, 8)] = float4(reflContrib, reflAlpha);
    }

    float3 accumulation = output_primary + output_indirect + sunDirect + reflContrib;

    //==================== TEMP DEBUG: nadir-ring localisation, mode 2 ===========
    //False-colours the path-traced mesh radiance so we can see which term the
    //ring lives in:  RED = indirect / GI,  GREEN = reflection,  BLUE = direct.
    //Report which channel(s) the ring appears in. Remove once the cause found.
    #if ATM_DEBUG_RING == 2
    {
        float dr = 1.0f - exp(-max(Luma(dbgRawIndirect), 0.0f) * 3.0f);
        float dg = 1.0f - exp(-max(Luma(dbgRawRefl),     0.0f) * 3.0f);
        float db = 1.0f - exp(-max(Luma(dbgRawPrimary) + Luma(dbgRawSun), 0.0f) * 3.0f);
        accumulation = float3(dr, dg, db);
    }
    #endif
    //============================================================================

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
            g_dlssDepth[DTid.xy] = DLSS_LinearDepthFromWorldPos(emPos);

            float2 curPix = DTid.xy;
            //DoF aware MV, pinhole projection on both ends so the lens offset
            //(curPix is the lens jittered pixel, emPos is the lens jittered hit) cancels out
            float2 prevPix = GetLastFramePixelCoordinates_Unclamped(
                emPos, prevView, prevProjection, dims, emInstID);
            float2 curPinholePix = GetCurrentFramePixelCoordinates_Unclamped(
                emPos, view, projection, dims, emInstID);
            bool validPrev = (prevPix.x > -1e8f) && (curPinholePix.x > -1e8f);
            float2 emMV = validPrev ? (prevPix - curPinholePix) : float2(0, 0);
            g_dlssMVec[curPix] = emMV;
            biasMV = emMV;
            isEmitterSurface = true;

            //shading normal for DLSS RR edge detection
            float3 emNormal = load_n1_s_with_instID(g_sample_current, pixelIdx, emInstID);
            g_dlssNormals[DTid.xy] = float4(emNormal, 0.0f);
        }
        else
        {
            //====================================
            //SKY OR CLOUD-COVERED SKY
            //====================================
            //Pass_clouds_primary_v8 wrote the transmittance-weighted cloud
            //hit position to scratch slot 12 (xyz = world pos, w = dist in
            //km). If the cloud has non-trivial opacity we feed cloud data
            //into DLSS RR instead of treating this as pure infinity — that
            //gives clouds proper camera-translation motion vectors, real
            //depth for edge detection, and a stable billboard normal.
            //Otherwise fall through to the rotation-only sky path.
            const float4 cloudHitData    = gScratchPing[uint3(DTid.xy, 12)];
            const float3 cloudHitPos     = cloudHitData.xyz;
            const float  cloudHitDistKm  = cloudHitData.w;
            const float3 cloudTr         = gScratchPing[uint3(DTid.xy, 11)].rgb;
            const float  cloudOpacity    = saturate(1.0f
                                         - dot(cloudTr, float3(0.2126f, 0.7152f, 0.0722f)));
            const bool   cloudIsDominant = (cloudHitDistKm > 0.0f) && (cloudOpacity > 0.05f);

            if (cloudIsDominant)
            {
                //Cloud-aware DLSS inputs.
                g_dlssDepth[DTid.xy] = DLSS_LinearDepthFromWorldPos(cloudHitPos);

                //World-space MV. Cloud is treated as world-static (no wind
                //animation contribution to MV), so only camera motion
                //matters. Parallax from camera translation is the bit that
                //the old rotation-only sky MV was missing — clouds smeared
                //heavily on dolly. This fixes it.
                float2 prevPix = GetLastFramePixelCoordinates_World(
                    cloudHitPos, prevView, prevProjection, dims);
                float2 curPinholePix = GetCurrentFramePixelCoordinates_World(
                    cloudHitPos, view, projection, dims);
                bool validPrev = (prevPix.x > -1e8f) && (curPinholePix.x > -1e8f);
                float2 cloudMV = validPrev ? (prevPix - curPinholePix) : float2(0, 0);
                g_dlssMVec[DTid.xy] = cloudMV;
                biasMV = cloudMV;

                //Cloud surface normal from Pass_clouds_primary_v8 (slot 13).
                //The cloud pass extracts the outward facing normal at the
                //transmittance weighted Hillaire mean depth by finite
                //differencing the macro shape density (no HF / cauliflower)
                //so neighbouring pixels of the same cumulus share roughly
                //matching normals — the spatial similarity signal RR uses
                //to pool samples across the cloud silhouette. Massive lift
                //over the previous camera facing billboard, which gave
                //every pixel of every cumulus the same normal and let RR
                //pool only through temporal accumulation (smearing /
                //boiling on dim or noisy clouds).
                //
                //Fall back to the billboard only when the cloud pass
                //couldn't produce a usable normal (zero vector) — happens
                //for thin clouds or when the camera sits inside a cumulus
                //where gradient extraction is ambiguous.
                const float3 cloudNormalFromPass = gScratchPing[uint3(DTid.xy, 13)].xyz;
                float3 cloudNormal;
                if (dot(cloudNormalFromPass, cloudNormalFromPass) > 0.25f)
                {
                    cloudNormal = cloudNormalFromPass;
                }
                else
                {
                    float2 nd = ((float2(DTid.xy) + 0.5f) / dims) * 2.0f - 1.0f;
                    float4 ndTarget = mul(projectionI, float4(nd.x, -nd.y, 1, 1));
                    cloudNormal = -normalize(mul(viewI, float4(ndTarget.xyz, 0)).xyz);
                }
                g_dlssNormals[DTid.xy] = float4(cloudNormal, 0.0f);
            }
            else
            {
                //Pure sky: clamp depth to cameraFar so DLSS RR's range
                //handling stays consistent.
                g_dlssDepth[DTid.xy] = cameraFar;
                g_dlssNormals[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);

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
                        skyMV = prevPix - float2(DTid.xy) - jitter;
                    }
                }
                g_dlssMVec[DTid.xy] = skyMV;
                biasMV = skyMV;
            }
        }

        biasInstID = emInstID;
        g_dlssSpecularAlbedo[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
        g_dlssDiffuseAlbedo[DTid.xy] = float4(1.0f, 1.0f, 1.0f, 0.0f);
        g_dlssRoughness[DTid.xy] = 1.0f;
        g_dlssSpecHitDist[DTid.xy] = hasPosition ? 0.0f : cameraFar;
        g_dlssSpecMVec[DTid.xy] = float2(0.0f, 0.0f);
        //Reinhard pre-tonemap (reversed in postprocess) so emitters survive AgX
        g_dlssInput[DTid.xy] = float4(DlssReinhard(accumulation), 1.0f);
        gOutput[uint3(DTid.xy, 5)] = float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
    else{
        //====================================
        //CLOUD-OCCLUDED MESH OVERRIDE
        //====================================
        //If a thick enough cloud sits between camera and mesh (typical
        //"above clouds, ground below" view), the visible surface for this
        //pixel is the cloud, not the mesh. Feeding DLSS RR the mesh's
        //depth/normal/MV in that case makes the upscaler resolve the noisy
        //mesh behind near-zero cloud transmittance — boiling and ghosting.
        //
        //Dominance is decided by the CLOUD-ONLY accumulated opacity that
        //cloud_primary stores in slot 13.w (1 - macro-shape transmittance,
        //atmospheric extinction excluded). Earlier attempts used either a
        //hard opacity threshold on combined atm+cloud Tr or a luma ratio
        //of cloudL vs mesh*Tr — both broke on long mesh rays because
        //cloudL bundles atmospheric in-scatter, so the see-through-cloud
        //case still registered as "cloud dominant" purely from haze.
        //
        //cloudAlpha > 0.5: cloud blocks more than half the light from beyond
        //  → cloud is the visible surface, DLSS tracks it
        //cloudAlpha < 0.5: cloud lets more than half through
        //  → mesh shows through, DLSS tracks the mesh
        const float4 cloudNormalAndAlpha = gScratchPing[uint3(DTid.xy, 13)];
        const float4 cloudHitDataM       = gScratchPing[uint3(DTid.xy, 12)];
        const float3 cloudHitPosM        = cloudHitDataM.xyz;
        const float  cloudHitDistKmM     = cloudHitDataM.w;
        const float  cloudAlphaM         = cloudNormalAndAlpha.w;
        const bool   cloudDominatesM     = (cloudHitDistKmM > 0.0f) && (cloudAlphaM > 0.5f);

        if (cloudDominatesM)
        {
            //Mirror of the sky-pixel cloud-dominant branch above. Cloud
            //depth, world-space cloud MV, gradient-extracted cloud normal
            //(billboard fallback), and the sky-style flat material (diffuse
            //white, no spec, full roughness) so DLSS RR resolves the cloud
            //silhouette rather than the mesh visible behind cloudTr ≈ 0.
            g_dlssDepth[DTid.xy] = DLSS_LinearDepthFromWorldPos(cloudHitPosM);

            float2 prevPixCM = GetLastFramePixelCoordinates_World(
                cloudHitPosM, prevView, prevProjection, dims);
            float2 curPinholePixCM = GetCurrentFramePixelCoordinates_World(
                cloudHitPosM, view, projection, dims);
            bool validPrevCM = (prevPixCM.x > -1e8f) && (curPinholePixCM.x > -1e8f);
            float2 cloudMVm = validPrevCM ? (prevPixCM - curPinholePixCM) : float2(0, 0);
            g_dlssMVec[DTid.xy] = cloudMVm;
            biasMV = cloudMVm;

            const float3 cloudNormalFromPassM = cloudNormalAndAlpha.xyz;
            float3 cloudNormalM;
            if (dot(cloudNormalFromPassM, cloudNormalFromPassM) > 0.25f)
            {
                cloudNormalM = cloudNormalFromPassM;
            }
            else
            {
                float2 ndM = ((float2(DTid.xy) + 0.5f) / dims) * 2.0f - 1.0f;
                float4 ndTargetM = mul(projectionI, float4(ndM.x, -ndM.y, 1, 1));
                cloudNormalM = -normalize(mul(viewI, float4(ndTargetM.xyz, 0)).xyz);
            }
            g_dlssNormals[DTid.xy] = float4(cloudNormalM, 0.0f);

            g_dlssDiffuseAlbedo[DTid.xy]  = float4(1.0f, 1.0f, 1.0f, 0.0f);
            g_dlssSpecularAlbedo[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);
            g_dlssRoughness[DTid.xy]      = 1.0f;
            g_dlssSpecHitDist[DTid.xy]    = 0.0f;
            g_dlssSpecMVec[DTid.xy]       = float2(0.0f, 0.0f);
            gOutput[uint3(DTid.xy, 5)]    = float4(1.0f, 1.0f, 1.0f, 1.0f);

            //Use the mesh's instID for the disocclusion check, NOT the sky
            //sentinel. g_sample_last still stores raygen's mesh instID
            //(this override only affects the local biasInstID this frame),
            //so a sentinel here would mismatch the prev sample every frame
            //of a stable cloud-over-mesh view → bias=1 permanently → DLSS
            //refuses to temporally accumulate the cloud and it stays noisy.
            //Cloud↔mesh transition disocclusion is handled by DLSS RR's
            //internal depth-disocclusion logic (cloud depth ≠ mesh depth).
            biasInstID = load_instID(g_sample_current, pixelIdx);

            g_dlssInput[DTid.xy] = float4(DlssReinhard(accumulation), 1.0f);
        }
        else
        {
            //reconstruct surface for DLSS from the baked G-buffer - no
            //per-triangle data, identical path for terrain and scene meshes.
            uint sInstID = load_instID(g_sample_current, pixelIdx);
            float3 sPos  = load_x1(g_sample_current, pixelIdx);
            SurfaceVertex sv = BuildVertex(g_sample_current, pixelIdx, sPos, camPosWorld);
            //DLSS RR input data
            g_dlssDepth[DTid.xy] = DLSS_LinearDepthFromWorldPos(sv.x);

            g_dlssNormals[DTid.xy] = float4(sv.n_s, 0.0f);
            g_dlssDiffuseAlbedo[DTid.xy] = float4(sv.Kd, 1.0f);
            g_dlssRoughness[DTid.xy] = sv.Pr;
            //debug mirror of diffuse albedo passed to DLSS RR
            gOutput[uint3(DTid.xy, 5)] = float4(sv.Kd, 1.0f);

            float2 curPix = DTid.xy;
            //DoF aware MV, sv.x is the lens jittered hit so its pinhole projection differs
            //from curPix. Use the pinhole projection of sv.x on both ends so the MV describes
            //pure scene motion in pinhole space, the lens offset cancels out
            float2 prevPix, curPinholePix;
            //PLANET: terrain is world-static and has no instanceProps entry -
            //use the world-static reprojection (skips the instance lookup).
            if (IsTerrainInstance(sInstID))
            {
                prevPix       = GetLastFramePixelCoordinates_World(sv.x, prevView, prevProjection, dims);
                curPinholePix = GetCurrentFramePixelCoordinates_World(sv.x, view, projection, dims);
            }
            else
            {
                prevPix       = GetLastFramePixelCoordinates_Unclamped(sv.x, prevView, prevProjection, dims, sInstID);
                curPinholePix = GetCurrentFramePixelCoordinates_Unclamped(sv.x, view, projection, dims, sInstID);
            }

            bool validPrev = (prevPix.x > -1e8f) && (curPinholePix.x > -1e8f);

            float2 mvPixels = validPrev ? (prevPix - curPinholePix) : float2(0.0, 0.0);

            g_dlssMVec[curPix] = mvPixels;

            biasInstID = sInstID;
            biasMV = mvPixels;

            //specular albedo
            float3 specularAlbedo = EnvBRDFApprox2(sv.Kd, sv.Pr, sv.Pm, dot(sv.o, sv.n_s));
            g_dlssSpecularAlbedo[DTid.xy] = float4(specularAlbedo, 0.0f);

            Reservoir rdi = loadReservoir(g_Reservoirs_current, pixelIdx);
            g_dlssSpecHitDist[DTid.xy] = length(rdi.x2 - sv.x);

            //specular MV from raygen's perfect reflection probe in scratch slice 4
            float specularity = Luma(specularAlbedo);
            float2 specMV = mvPixels;
            bool validSpecReproj = false;
            {
                float4 reflData = gScratchPing[uint3(DTid.xy, 4)];
                uint   reflInstID = asuint(reflData.w);
                //PLANET: terrain instIDs excluded - rough terrain doesn't use
                //the specular reprojection, and it has no instanceProps entry.
                if (specularity > 0.04f && reflInstID < TERRAIN_INSTANCE_BASE)
                {
                    //DoF aware spec MV, pinhole on both ends like the surface MV above
                    float2 prevRefl = GetLastFramePixelCoordinates_Unclamped(
                        reflData.xyz, prevView, prevProjection, dims, reflInstID);
                    float2 curRefl  = GetCurrentFramePixelCoordinates_Unclamped(
                        reflData.xyz, view, projection, dims, reflInstID);
                    bool validRefl = (prevRefl.x > -1e8f) && (curRefl.x > -1e8f);
                    if (validRefl)
                    {
                        specMV = prevRefl - curRefl;
                        validSpecReproj = true;
                    }
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
            g_dlssInput[DTid.xy] = float4(DlssReinhard(accumulation), 1.0f);
#endif
        }
    }

    //====================================
    //BIAS HINT INSTANCE ID DISOCCLUSION
    //====================================
    //instID mismatch at reprojected pixel marks disocclusion, bias=1
    {
        float disoccBias = 1.0f;
        float2 reprojPrev = float2(DTid.xy) + biasMV;
        int2 prevPixI = int2(round(reprojPrev));
        if (all(prevPixI >= 0) && all(prevPixI < int2(dims))) {
            uint prevPixelIdx = MapPixelID(uint2(dims), prevPixI);
            uint prevInstID = load_instID(g_sample_last, prevPixelIdx);
            disoccBias = (biasInstID != prevInstID) ? 1.0f : 0.0f;
        }
        float emitterBias = isEmitterSurface ? 1.0f : 0.0f;

        //reflection dominated pixels lean toward the input so edges stay crisp
        float reflectionBias = 0.0f;
        {
            const float reflLuma  = Luma(reflContrib);
            const float totalLuma = Luma(accumulation);
            if (totalLuma > 1e-5f && reflLuma > 1e-5f) {
                reflectionBias = saturate(reflLuma / totalLuma);
            }
        }

        float bias = max(disoccBias, max(emitterBias, reflectionBias));
        g_dlssBiasHint[DTid.xy] = bias;
        if (disoccBias > 0.5f && !isEmissiveOrSky) {
            g_dlssSpecHitDist[DTid.xy] = g_dlssDepth[DTid.xy];
        }
    }

    //====================================
    //DLSS TRANSPARENCY OVERLAY HOOK
    //====================================
    //placeholder side write, host can wire as a post denoise overlay later
    g_dlssTransparency[DTid.xy] = float4(reflContrib, 0.0f);
}
