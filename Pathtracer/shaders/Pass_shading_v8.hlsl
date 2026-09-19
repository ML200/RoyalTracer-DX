#define COMPUTE_PASS
#include "Includes_v8.hlsli"

inline float3 DlssReinhard(float3 c) {
    c = max(c, 0.0f);
    const float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    return c / (1.0f + lum);
}

#define DLSS_PT_INPUT_LUMA_CAP 64.0f

inline float ReadExposureForCap() {
    const float AE_KEY_VALUE = 0.18f;
    const float smoothedLog2Lum = asfloat(gAutoExpose.Load(AE_OFFS_SMOOTHED));
    return AE_KEY_VALUE / max(exp2(smoothedLog2Lum), 1e-6f);
}

inline float3 ScrubNonFiniteIn(float3 c) {
    return (any(isnan(c)) || any(isinf(c))) ? float3(0, 0, 0) : c;
}

inline float3 DlssEncode(float3 c) {
    c = ScrubNonFiniteIn(c);
    if (PT_ONLY_MODE) {

        return max(c, 0.0f) * ReadExposureForCap();
    }
    return DlssReinhard(c);
}

#define DLSS_EMITTER_CAP 16.0f

#define DLSS_SPEC_ROUGHNESS_THRESHOLD 0.25f
inline float3 ClampEmitterLum(float3 c) {
    const float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    return (lum > DLSS_EMITTER_CAP) ? c * (DLSS_EMITTER_CAP / lum) : c;
}

// Legacy base surface: the primary hit, or the surface seen straight through thin glass panes.
struct RRGuide { float3 x; float3 n; float3 Kd; float Pr; float Pm; uint instID; };

inline RRGuide ResolveRRGuideThroughGlass(SurfaceVertex sv, uint sInstID, float3 camPos)
{
    RRGuide g;
    g.x = sv.x; g.n = sv.n_s; g.Kd = sv.Kd; g.Pr = sv.Pr; g.Pm = sv.Pm; g.instID = sInstID;

    if (!LoadIsThinGlass(sv.matID))
        return g;

    const float3 vdir = normalize(sv.x - camPos);
    float3 tint = LoadTf(sv.matID);
    float3 ro   = offset_ray(sv.x, -sv.n_s);

    [loop]
    for (uint pane = 0u; pane < 16u; ++pane)
    {
        RayDesc r;
        r.Origin    = ro;
        r.Direction = vdir;
        r.TMin      = 0.00001f;
        r.TMax      = RAY_TMAX_PLANET;

        RayQuery<RAY_FLAG_NONE, RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS> q;
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

        HitInfo bh = EvalSurfaceState(hi, hp, q.CommittedTriangleBarycentrics(), ro, 0u);
        float3 bKd; float bPr, bPm;
        RefetchMaterial(hm, bh.uv, bKd, bPr, bPm, 0u);
        g.x = bh.hitPos; g.n = bh.hitNormal; g.Kd = bKd * tint; g.Pr = bPr; g.Pm = bPm;
        g.instID = hi;
        return g;
    }
    return g;
}

// Guides before primary surface replacement: everything describes the reflector, except that
// thin glass takes depth, motion and albedo from the surface behind it.
void WriteLegacyGuides(uint2 px, uint pixelIdx, float2 dims, float3 camPos)
{
    const uint   sInstID = load_instID(g_sample_current, pixelIdx);
    const float3 sPos    = load_x1(g_sample_current, pixelIdx);
    const SurfaceVertex sv = BuildVertex(g_sample_current, pixelIdx, sPos, camPos);

    const RRGuide rg = ResolveRRGuideThroughGlass(sv, sInstID, camPos);

    g_dlssDepth[px] = DLSS_GuideDepthFromWorldPos(rg.x);

    const float3 specularAlbedo = EnvBRDFApprox2(sv.Kd, sv.Pr, sv.Pm, dot(sv.o, sv.n_s));
    const float  reflW          = saturate(Luma(specularAlbedo));

    g_dlssNormals[px] = float4(sv.n_s, sv.Pr);
    g_dlssDiffuseAlbedo[px] = float4(rg.Kd, 1.0f);
    g_dlssRoughness[px] = sv.Pr;
#if SHADING_DEBUG_SLICES
    gOutput[uint3(px, 5)] = float4(rg.Kd, 1.0f);
#endif

    const float2 mvPixels = SurfaceMotionVector(px, dims, rg.x, rg.instID);
    g_dlssMVec[px] = mvPixels;
    g_dlssSpecularAlbedo[px] = float4(specularAlbedo, 0.0f);

    const float4 reflData   = gScratchPing[uint3(px, 4)];
    const uint   reflInstID = asuint(reflData.w);
    g_dlssSpecHitDist[px] = (reflInstID != 0xFFFFFFFFu)
        ? min(length(reflData.xyz - sv.x), DLSS_SPEC_HIT_MAX)
        : DLSS_SPEC_HIT_MAX;

    float2 specMV = LoadIsThinGlass(sv.matID) ? SurfaceMotionVector(px, dims, sv.x, sInstID) : mvPixels;
    if (reflW > 0.04f && sv.Pr < DLSS_SPEC_ROUGHNESS_THRESHOLD && reflInstID != 0xFFFFFFFFu)
    {
        const float2 prevRefl = GetLastFramePixelCoordinates_Unclamped(reflData.xyz, prevView, prevProjection, dims, reflInstID);
        const float2 curRefl  = GetCurrentFramePixelCoordinates_Unclamped(reflData.xyz, view, projection, dims, reflInstID);
        if (prevRefl.x > -1e8f && curRefl.x > -1e8f)
            specMV = prevRefl - curRefl;
    }
    g_dlssSpecMVec[px] = specMV;
}

// Primary surface replacement. The pixel's energy is split into the mirror image of the near-delta
// lobes and the part transmitted or scattered by the base surface. Both ends come from delta-chain
// walks that stop at the first diffuse or rough surface, and every guide blends between the two by
// their reflectance shares. Returns the bias hint.
float WritePsrGuides(uint2 px, uint pixelIdx, float2 dims, float3 camPos)
{
    const uint   sInstID = load_instID(g_sample_current, pixelIdx);
    const float3 sPos    = load_x1(g_sample_current, pixelIdx);
    const SurfaceVertex sv = BuildVertex(g_sample_current, pixelIdx, sPos, camPos);
    const float NoV        = saturate(dot(sv.o, sv.n_s));
    const bool  thin       = LoadIsThinGlass(sv.matID);
    const bool  seeThrough = PsrSeeThroughMaterial(sv.matID);

    // The mirror share is the material Fresnel toward the camera, not the sampler's pick probability.
    float3 mirror;            // energy share of the mirror image
    float3 residual;          // near-delta lobe energy the probe does not represent (roughness fade)
    float3 baseT;             // energy share reaching the base surface
    float  mirrorRoughness;   // roughness of the lobe forming the mirror image
    PsrChainEnd base = PsrChainSurface(sv.x, sv.n_s, sv.Kd, sv.Pr, sv.Pm, sInstID);
    if (seeThrough)
    {
        // Thin glass passes straight through, clear glass refracts; the pane reflection is the mirror.
        const float3 F = thin ? FresnelDielectric(sv.o, sv.n_s, sv.etai, sv.etat)
                              : FresnelDielectricTIR(sv.o, sv.n_s, sv.etai, sv.etat);
        float3 dir = -sv.o;
        const bool  passes = thin || RefractVector(sv.o, sv.n_s, sv.etai / sv.etat, dir);
        const float fade   = PsrRoughnessFade(sv.Pr);
        mirror          = F * fade;
        residual        = F * (1.0f - fade);
        mirrorRoughness = sv.Pr;
        baseT = passes ? (1.0f - F) * (thin ? LoadTf(sv.matID) : 1.0f) : 0.0f;
        // Until the chain reports a surface, the base is the pane itself: rough glass shows a blur on it.
        base = PsrChainSurface(sv.x, sv.n_s, float3(1.0f, 1.0f, 1.0f), sv.Pr, 0.0f, sInstID);
        if (passes && fade > 0.0f)
        {
            const bool enters = !thin && !load_backface(g_sample_current, pixelIdx);
            base = PsrWalkDeltaChain(offset_ray(sv.x, -sv.n_s), dir, enters ? sv.matID : MEDIUM_INVALID,
                                     PsrIdentity(), sv.x, -sv.o, true, DLSS_PSR_MAX_CHAIN, false);
            baseT *= base.throughput;
        }
    }
    else
    {
        // Coat over GGX over diffuse; each lower layer receives what the upper one does not reflect.
        const bool   psrLobes = PsrCandidateMaterial(sv.matID, sv.Pr);
        const float  pc       = LoadPc(sv.matID);
        const float  pcr      = LoadPcr(sv.matID);
        const float3 coatR    = (pc > 0.0f) ? pc * GGXDirectionalReflectance(pcr, NoV, sv.etai, sv.etat, true) : 0.0f;
        const float3 ggxR     = (1.0f - coatR) * EnvBRDFApprox2(sv.Kd, sv.Pr, sv.Pm, NoV);
        const float  coatFade = (psrLobes && pc > 0.0f) ? PsrRoughnessFade(pcr) : 0.0f;
        const float  ggxFade  = psrLobes ? PsrRoughnessFade(sv.Pr) : 0.0f;
        mirror          = coatR * coatFade + ggxR * ggxFade;
        residual        = coatR * (1.0f - coatFade) + ggxR * (1.0f - ggxFade);
        mirrorRoughness = (coatFade > 0.0f && Luma(coatR) * coatFade >= Luma(ggxR) * ggxFade) ? pcr : sv.Pr;
        baseT           = saturate(1.0f - coatR - ggxR);
    }

    // The specular albedo guide is always the primary surface's own specular share: it tells RR how
    // much of the pixel follows the specular motion, so it is never replaced or attenuated by the chain.
    const float3 primarySpecular = mirror + residual;

    // Everything reaching the base surface is treated as its diffuse content, including the rough
    // specular of a surface seen through glass.
    const bool   baseSky     = (base.flags & DLSS_PSR_FLAG_SKY) != 0u;
    const bool   baseEmitter = (base.flags & DLSS_PSR_FLAG_EMITTER) != 0u;
    const float3 baseAlbedo  = (seeThrough && !baseSky && !baseEmitter)
        ? base.Kd * (1.0f - base.Pm) + EnvBRDFApprox2(base.Kd, base.Pr, base.Pm, saturate(dot(base.nVirtual, sv.o)))
        : base.Kd * (1.0f - base.Pm);
    const float3 baseDiffuse = baseT * baseAlbedo;

    // Virtual surface: the end of the reflection chain, mirrored into the primary view. The plain
    // first reflection hit stays the specular layer's own hit.
    const float4   reflFirst  = gScratchPing[uint3(px, 4)];
    const PsrProbe probe      = PsrProbeUnpack(gScratchPing[uint3(px, DLSS_PSR_PROBE_SLOT)]);
    const bool     probeValid = (probe.flags & DLSS_PSR_FLAG_VALID) != 0u;
    const float4   reflData   = probeValid ? gScratchPing[uint3(px, DLSS_PSR_CHAIN_SLOT)] : reflFirst;
    const uint     reflInstID = asuint(reflData.w);
    if (probeValid) mirror *= probe.throughput;
    else { residual += mirror; mirror = 0.0f; }

    const bool   virtualSky     = reflInstID == 0xFFFFFFFFu;
    const bool   virtualEmitter = (probe.flags & DLSS_PSR_FLAG_EMITTER) != 0u;
    const float3 nV  = virtualSky ? sv.o : probe.nVirtual;
    const float3 KdV = virtualSky ? float3(1.0f, 1.0f, 1.0f) : probe.Kd;
    const float  PrV = virtualSky ? 1.0f : probe.Pr;
    const float  PmV = virtualSky ? 0.0f : probe.Pm;
    const float3 xV  = virtualSky ? camPos - sv.o * cameraFar : reflData.xyz;
    const float3 diffV = KdV * (1.0f - PmV);

    // Every merged guide follows the mirror's Fresnel share of the pixel's reflectance: the delta part
    // of the lobe, attenuated by the chain, against the base surface and the rough residual.
    const float mirrorShare = Luma(mirror);
    const float w = mirrorShare / max(mirrorShare + Luma(baseDiffuse + residual), 1e-4f);

    // Diffuse albedo takes the mirror image in proportion to its ownership of the pixel.
    const float3 diffuseAlbedo  = baseDiffuse + mirror * diffV * w;
    const float3 specularAlbedo = primarySpecular;

    // Glass shows the surface behind it; an opaque reflector shows its own mirror lobe.
    const float baseRoughness = seeThrough ? base.Pr : mirrorRoughness;
    const float roughness     = lerp(baseRoughness, PrV, w);

    float3 n = lerp(base.nVirtual, nV, w);
    n = (dot(n, n) > 1e-6f) ? normalize(n) : (w > 0.5f ? nV : base.nVirtual);

    // Depth blends in the encoded reciprocal domain, so a distant mirror image only nudges a near surface.
    const float depth = lerp(DLSS_GuideDepthFromWorldPos(base.xVirtual), DLSS_GuideDepthFromWorldPos(xV), w);

    g_dlssDepth[px]          = depth;
    g_dlssNormals[px]        = float4(n, roughness);
    g_dlssRoughness[px]      = roughness;
    g_dlssDiffuseAlbedo[px]  = float4(diffuseAlbedo, 1.0f);
    g_dlssSpecularAlbedo[px] = float4(specularAlbedo, 0.0f);
#if SHADING_DEBUG_SLICES
    gOutput[uint3(px, 5)] = float4(diffuseAlbedo, 1.0f);
#endif

    // Base motion follows the chain end through the chain; a single primary reflection is re-mirrored
    // exactly, longer reflection chains move the virtual point with the end surface's instance.
    const float2 baseMV    = PsrChainMotionVector(px, dims, base);
    float2 virtualMV = virtualSky ? SkyMotionVector(px, dims)
        : (probe.bounces <= 1u
            ? PsrVirtualMotionVector(reflData.xyz, reflInstID, sv.x, sv.n_s, sInstID, dims)
            : SurfaceMotionVector(px, dims, reflData.xyz, reflInstID));
    // A pane ending the reflection chain: its transmission moves with the pane, its reflection of
    // distant surroundings appears at infinity along the primary ray and moves like the sky.
    if ((probe.flags & DLSS_PSR_FLAG_GLASS) != 0u)
        virtualMV = lerp(virtualMV, SkyMotionVector(px, dims), probe.paneF);

    // Diffuse motion on opaque surfaces follows the same share; DLSS_PSR_MV_MIX moves it from a hard
    // hand-over at half toward a proportional blend. Glass keeps the transmission target's motion
    // unblended. The specular channel always follows the mirror image while the lobe is sharp.
    const float mvWeight = lerp(w > 0.5f ? 1.0f : 0.0f, w, DLSS_PSR_MV_MIX);
    const bool  mvBlend  = !seeThrough && (dbg_dlssLayer & DLSS_GUIDE_OPT_NO_MV_BLEND) == 0u;
    g_dlssMVec[px] = mvBlend ? lerp(baseMV, virtualMV, mvWeight) : baseMV;

    const bool   mirrorLayer = mirrorRoughness < DLSS_SPEC_ROUGHNESS_THRESHOLD && Luma(mirror + residual) > 0.02f;
    const float2 surfaceMV   = seeThrough ? SurfaceMotionVector(px, dims, sv.x, sInstID) : baseMV;
    g_dlssSpecMVec[px]    = mirrorLayer ? virtualMV : surfaceMV;
    g_dlssSpecHitDist[px] = (asuint(reflFirst.w) != 0xFFFFFFFFu)
        ? min(length(reflFirst.xyz - sv.x), DLSS_SPEC_HIT_MAX) : DLSS_SPEC_HIT_MAX;

    // Emitters reached through delta chains are as deterministic as emitters seen directly.
    float bias = 0.0f;
    if (virtualEmitter && mirrorRoughness < SMOOTH_SPECULAR_THRESHOLD && (probe.flags & DLSS_PSR_FLAG_DELTA) != 0u)
        bias += w;
    if (seeThrough && baseEmitter && sv.Pr < SMOOTH_SPECULAR_THRESHOLD && (base.flags & DLSS_PSR_FLAG_DELTA) != 0u)
        bias += 1.0f - w;
    return saturate(bias);
}

#include "CumulusGuides_v8.hlsli"

[numthreads(16, 16, 1)]
// Resolve hit state, visibility, and primary shading outputs.
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;

    const float3 camPosWorld = mul(viewI, float4(0, 0, 0, 1)).xyz;

    float3 output_primary  = gScratchPing[uint3(DTid.xy, 1)].rgb;
    float3 output_indirect = gScratchPing[uint3(DTid.xy, 2)].rgb;

    const float3 dbgRawPrimary  = output_primary;
    const float3 dbgRawIndirect = output_indirect;

    const float3 atmosphereL = gScratchPing[uint3(DTid.xy, 10)].rgb;
    const float3 atmosphereTr = gScratchPing[uint3(DTid.xy, 11)].rgb;

    float3 accumulation = (output_primary + output_indirect) * atmosphereTr + atmosphereL;

#if SHADING_DEBUG_SLICES
    gScratchPing[uint3(DTid.xy, 1)] = float4(accumulation, 0);
#endif

    #if ATM_DEBUG_RING == 2
    {
        float dr = 1.0f - exp(-max(Luma(dbgRawIndirect), 0.0f) * 3.0f);
        float dg = 0.0f;
        float db = 1.0f - exp(-max(Luma(dbgRawPrimary), 0.0f) * 3.0f);
        accumulation = float3(dr, dg, db);
    }
    #endif

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

    bool  isEmitterSurface = false;
    float psrBias = 0.0f;

    bool isEmissiveOrSky = load_isEmitter(g_sample_current, pixelIdx);
    if (isEmissiveOrSky)
    {

        uint emInstID = load_instID(g_sample_current, pixelIdx);
        bool hasPosition = (emInstID != 0xFFFFFFFFu);

        if (hasPosition)
        {

            float3 emPos  = load_x1(g_sample_current, pixelIdx);
            g_dlssDepth[DTid.xy] = DLSS_GuideDepthFromWorldPos(emPos);
            g_dlssMVec[DTid.xy] = SurfaceMotionVector(DTid.xy, dims, emPos, emInstID);
            isEmitterSurface = true;

            const float3 emNormal = load_n1_s_with_instID(g_sample_current, pixelIdx, emInstID);

            g_dlssNormals[DTid.xy] = float4(emNormal, 1.0f);
        }
        else
        {
            g_dlssDepth[DTid.xy] = 0.0f;
            g_dlssNormals[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 1.0f);
            g_dlssMVec[DTid.xy] = SkyMotionVector(DTid.xy, dims);
        }

        g_dlssSpecularAlbedo[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 0.0f);

        const float3 emitterAlbedo = saturate(ScrubNonFiniteIn(dbgRawPrimary));
        g_dlssDiffuseAlbedo[DTid.xy] = float4(hasPosition ? emitterAlbedo : float3(0.5f, 0.5f, 0.5f), 0.0f);
        g_dlssRoughness[DTid.xy] = 1.0f;

        g_dlssSpecHitDist[DTid.xy] = hasPosition ? 0.0f : DLSS_SPEC_HIT_MAX;
        g_dlssSpecMVec[DTid.xy] = float2(0.0f, 0.0f);

        float3 emitterRadiance = (CLAMP_EMITTERS_MODE && hasPosition)
                                    ? ClampEmitterLum(accumulation)
                                    : accumulation;
        float3 emitterInput = DlssEncode(emitterRadiance);
        if (PT_ONLY_MODE && hasPosition) {
            const float lum = dot(emitterInput, float3(0.2126f, 0.7152f, 0.0722f));
            if (lum > DLSS_PT_INPUT_LUMA_CAP)
                emitterInput *= DLSS_PT_INPUT_LUMA_CAP / lum;
        }
        g_dlssInput[DTid.xy] = float4(emitterInput, 1.0f);
#if SHADING_DEBUG_SLICES
        gOutput[uint3(DTid.xy, 5)] = float4(1.0f, 1.0f, 1.0f, 1.0f);
#endif
    }
    else{
        if ((dbg_dlssLayer & DLSS_GUIDE_OPT_NO_PSR) != 0u)
            WriteLegacyGuides(DTid.xy, pixelIdx, dims, camPosWorld);
        else
            psrBias = WritePsrGuides(DTid.xy, pixelIdx, dims, camPosWorld);

#if ATM_DEBUG_RING == 4

        g_dlssInput[DTid.xy] = float4(g_dlssNormals[DTid.xy].xyz * 0.5f + 0.5f, 1.0f);
#else
        g_dlssInput[DTid.xy] = float4(DlssEncode(accumulation), 1.0f);
#endif

    }

    ApplyCumulusGuides(DTid.xy,pixelIdx,camPosWorld);
    float cloudOpacity=gScratchPing[uint3(DTid.xy,CUMULUS_NORMAL_SLOT)].w;
    g_dlssBiasHint[DTid.xy] = (isEmitterSurface ? 1.0f : psrBias) * (1.0f-cloudOpacity);

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

        if (bad != 0u)
            gAutoExpose.InterlockedCompareStore(SENT_OFFS_FIRSTBAD, 0u,
                                                ((DTid.y + 1u) << 16) | (DTid.x + 1u));

        const bool  nearCap = PT_ONLY_MODE && isEmitterSurface &&
                              (sLum >= DLSS_PT_INPUT_LUMA_CAP * 0.999f);
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
