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
//AERIAL PERSPECTIVE HELPER
//====================================
//Atmospheric extinction + in-scatter between camera and a scene hit point.
//Returns the modulated scene radiance. Distant outdoor surfaces pick up the
//haze tint, indoor / close geometry passes through nearly unchanged. Skipped
//for sky/sun pixels (those already integrate the full atmosphere in raygen).
inline float3 ApplyAerialPerspective(float3 sceneRadiance, float3 camPosWorld,
                                     float3 hitPosWorld, float3 sunDir)
{
    const float3 toHit     = hitPosWorld - camPosWorld;
    const float  worldDist = length(toHit);
    if (worldDist <= 1e-3f) return sceneRadiance;

    const float3 viewDir   = toHit / worldDist;
    const float  hitDistKm = worldDist * (1.0f / WORLD_UNITS_PER_KM);

    float3 transmittance;
    const float3 inScatter = ComputeAerialPerspective(viewDir, sunDir, hitDistKm, transmittance);

    return sceneRadiance * transmittance + inScatter * SKY_INTENSITY;
}

//====================================
//SHADING PASS
//====================================
[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;

    //Floating origin: viewI returns the *shifted* camera (view matrix
    //built in floating origin space). Most downstream math (aerial
    //perspective distance, reflections, etc.) lives in shifted world too
    //because hit positions are reconstructed via the shifted instance
    //transforms — keep camPosWorld in that shifted frame. Atmosphere /
    //planet observer math is the one exception: it needs the absolute
    //altitude above the planet centre, so add sceneOriginWorld back just
    //for SetSkyObserver.
    const float3 camPosWorld    = mul(viewI, float4(0, 0, 0, 1)).xyz;
    const float3 camPosAbsWorld = camPosWorld + sceneOriginWorld;
    SetSkyObserver(camPosAbsWorld);
    const SunState sunState = ComputeSunState();

    float3 output_primary  = gScratchPing[uint3(DTid.xy, 1)].rgb;
    float3 output_indirect = gScratchPing[uint3(DTid.xy, 2)].rgb;
    float3 sunDirect       = gScratchPing[uint3(DTid.xy, 3)].rgb;

    //====================================
    //PRIMARY CLOUD COMPOSITE (SKY PIXELS)
    //====================================
    //Pass_clouds_primary_v8 wrote cloudL into slot 10 and cloudTr into
    //slot 11 for every sky pixel, identity (0, 1) for the rest. Compose
    //here so the primary slots feed both the DLSS RR input AND the
    //postprocess noisy/gt debug paths with clouds already merged in. We
    //write the composited values back to slots 1 & 2 so postprocess
    //(which does its own sum of slot1+slot2+slot3) picks them up
    //without needing changes there.
    //
    //Branchless composite on the identity values: on non-sky pixels
    //cloudTr=(1,1,1) and cloudL=(0,0,0), so output_primary * cloudTr +
    //cloudL == output_primary — no behavioural change for surface hits.
    {
        const float2 dimsCO    = float2(IMG_W, IMG_H);
        const uint   pixelIdxCO = MapPixelID(dimsCO, DTid.xy);
        const uint   coInst     = load_instID(g_sample_current, pixelIdxCO);
        if (coInst == 0xFFFFFFFFu)
        {
            const float3 cloudL  = gScratchPing[uint3(DTid.xy, 10)].rgb;
            const float3 cloudTr = gScratchPing[uint3(DTid.xy, 11)].rgb;
            output_primary  = output_primary  * cloudTr + cloudL;
            output_indirect = output_indirect * cloudTr + cloudL;
            gScratchPing[uint3(DTid.xy, 1)] = float4(output_primary,  0);
            gScratchPing[uint3(DTid.xy, 2)] = float4(output_indirect, 0);
        }
    }

    //====================================
    //X1 SHARP REFLECTION CONTRIBUTION
    //====================================
    //slot 8 is the durable handoff, debug passes scribble g_NrcInferenceOut later
    float3 reflContrib = float3(0, 0, 0);
    {
        const float4 reflPack = gScratchPing[uint3(DTid.xy, 8)];
        if (reflPack.w > 0.0f)
            reflContrib = max(float3(0, 0, 0), reflPack.rgb);
    }

    float3 accumulation = output_primary + output_indirect + sunDirect + reflContrib;

    //====================================
    //AERIAL PERSPECTIVE
    //====================================
    //Apply atmosphere between camera and the primary hit point before the running
    //average + DLSS bound so every downstream consumer (gPermanentData ground
    //truth slice, the DLSS clean slice, the noisy debug slice) sees the same
    //atmospherically modulated radiance. Skipped for sky/sun rays — those are
    //integrated through the full atmosphere already inside the raygen miss path.
    {
        float2 dimsAP    = float2(IMG_W, IMG_H);
        uint   pixelIdxAP = MapPixelID(dimsAP, DTid.xy);
        bool isEmissiveOrSkyAP = load_isEmitter(g_sample_current, pixelIdxAP);
        uint apInstID = load_instID(g_sample_current, pixelIdxAP);
        bool hasPositionAP = (apInstID != 0xFFFFFFFFu);
        if (hasPositionAP) {
            uint   apPrimID  = load_primID(g_sample_current, pixelIdxAP);
            float2 apBary    = load_bary(g_sample_current, pixelIdxAP);
            float3 hitPosWorld = ReconstructPosition(apInstID, apPrimID, apBary);
            accumulation = ApplyAerialPerspective(accumulation, camPosWorld,
                                                  hitPosWorld, sunState.dirWS);
        }
    }

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
            uint emPrimID = load_primID(g_sample_current, pixelIdx);
            float2 emBary = load_bary(g_sample_current, pixelIdx);
            float3 emPos  = ReconstructPosition(emInstID, emPrimID, emBary);
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
            //sky, clamp to cameraFar to stay in DLSS RR range
            g_dlssDepth[DTid.xy] = cameraFar;
            g_dlssNormals[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);

            //Sky / planet body MV: rotation only reprojection. Sky is at
            //infinite distance so camera translation between frames must
            //not contribute to its apparent motion — only rotation does.
            //The previous "camPos + worldDir * cameraFar" trick breaks at
            //orbital flight speeds where camera translation per frame can
            //exceed cameraFar, making the far point arbitrarily close to
            //the camera and the MV dominated by translation instead of
            //rotation. Direction only reprojection sidesteps the issue.
            //
            //Math: transform worldDir into the previous view frame's
            //*rotation* (mul with w=0 ignores the translation row), then
            //project at any large positive view distance. The perspective
            //divide cancels the distance, so the exact value is arbitrary
            //as long as it doesn't underflow / overflow.
            float2 d = ((float2(DTid.xy) + 0.5f) / dims) * 2.0f - 1.0f;
            float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
            float3 worldDir = normalize(mul(viewI, float4(target.xyz, 0)).xyz);
            float3 prevViewDir = mul(prevView, float4(worldDir, 0)).xyz;
            float2 skyMV = float2(0.0f, 0.0f);
            //RH projection: in-front-of-camera has view-space z < 0
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
        //reconstruct surface for DLSS
        uint sInstID = load_instID(g_sample_current, pixelIdx);
        uint sPrimID = load_primID(g_sample_current, pixelIdx);
        float2 sBary = load_bary(g_sample_current, pixelIdx);
        SurfaceVertex sv = BuildVertex(sInstID, sPrimID, sBary, camPosWorld);
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
        float2 prevPix = GetLastFramePixelCoordinates_Unclamped(sv.x, prevView, prevProjection, dims, sInstID);
        float2 curPinholePix = GetCurrentFramePixelCoordinates_Unclamped(sv.x, view, projection, dims, sInstID);

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
            if (specularity > 0.04f && reflInstID < 0xFFFFFFFEu)
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
        g_dlssInput[DTid.xy] = float4(DlssReinhard(accumulation), 1.0f);
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
