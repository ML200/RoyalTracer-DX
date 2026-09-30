#include "Includes_v8.hlsli"



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
    return max(ScrubNonFiniteIn(c), 0.0f) * ReadExposureForCap();
}

inline float3 PixelViewDir(uint2 px, float2 dims) {
    const float2 d      = ((float2(px) + 0.5f) / dims) * 2.0f - 1.0f;
    const float4 target = mul(projectionI, float4(d.x, -d.y, 1, 1));
    return normalize(mul(viewI, float4(target.xyz, 0)).xyz);
}

// Luminance floor, not per channel: keeps a night sky's hue.
static const float kMinSkyGuideLuma = 1e-3f;

inline float3 SkyGuideAlbedo(float3 dir) {
    const float3 c   = DlssEncode(EvaluateSkyBackground(dir));
    const float  lum = Luma(c);
    if (lum >= kMinSkyGuideLuma)
        return saturate(c);
    return lum > 1e-8f ? saturate(c * (kMinSkyGuideLuma / lum)) : kMinSkyGuideLuma.xxx;
}

#define DLSS_EMITTER_CAP 16.0f

#define DLSS_SPEC_ROUGHNESS_THRESHOLD 0.25f

struct DlssGuides {
    float  depth;
    float2 mv;
    float3 n;
    float  roughness;
    float4 diffuseAlbedo;
    float3 specularAlbedo;
    float2 specMv;
    float  specHitDist;
};
inline float3 ClampEmitterLum(float3 c) {
    const float lum = 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
    return (lum > DLSS_EMITTER_CAP) ? c * (DLSS_EMITTER_CAP / lum) : c;
}

// Primary hit, or the surface behind thin glass.
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
    float  pathLength = length(sv.x - camPos);

    [loop]
    for (uint pane = 0u; pane < 16u; ++pane)
    {
        RayDesc r;
        r.Origin    = ro;
        r.Direction = vdir;
        r.TMin      = 0.00001f;
        r.TMax      = RAY_TMAX_PLANET;

        const bool traceable = IsRayDescValid(r);
        RayQuery<RAY_FLAG_NONE, RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS> q;
        if (traceable)
        {
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
        }

        if (!traceable || q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        {

            g.x = camPos + vdir * cameraFar; g.n = -vdir;
            // Sky behind the glass: its colour, not white.
            g.Kd = SkyGuideAlbedo(vdir); g.Pr = 1.0f; g.Pm = 0.0f;
            g.instID = 0xFFFFFFFFu;
            return g;
        }

        const uint   hi   = q.CommittedInstanceID();
        const uint   hp   = FlatPrimID(hi, q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex());
        const uint   hm   = GetMatIDFast(hi, hp);
        const float3 hpos = ro + vdir * q.CommittedRayT();
        pathLength += q.CommittedRayT();

        if (LoadIsThinGlass(hm))
        {
            tint *= LoadTf(hm);
            const float3 geoN = CandidateGeoNormalW(hi, hp);
            ro = offset_ray(hpos, (dot(vdir, geoN) >= 0.0f) ? geoN : -geoN);
            continue;
        }

        HitInfo bh = EvalSurfaceState(hi, hp, q.CommittedTriangleBarycentrics(), ro, PixelConeAngle() * pathLength);
        float3 bKd; float bPr, bPm;
        RefetchMaterial(hm, bh, bKd, bPr, bPm);
        g.x = bh.hitPos; g.n = bh.hitNormal; g.Kd = bKd * tint; g.Pr = bPr; g.Pm = bPm;
        g.instID = hi;
        return g;
    }
    return g;
}

// Bed seen through the water, for the albedo guide only.
struct OceanBelow { float3 albedo; float3 weight; bool hit; };

OceanBelow ResolveOceanBelow(SurfaceVertex sv, float3 transmitted, float3 camPos)
{
    OceanBelow b;
    b.albedo = 0.0f; b.weight = 0.0f; b.hit = false;
    if (!OceanMediumEnabled() || !any(transmitted > 0.01f))
        return b;

    // TIR: nothing below.
    const float3 vdir = normalize(sv.x - camPos);
    const float3 n = dot(vdir, sv.n_s) < 0.0f ? sv.n_s : -sv.n_s;
    const float3 dir = refract(vdir, n, sv.etai / sv.etat);
    if (dot(dir, dir) < 1e-6f)
        return b;

    RayDesc r;
    r.Origin    = offset_ray(sv.x, -n);
    r.Direction = dir;
    r.TMin      = 0.00001f;
    r.TMax      = RAY_TMAX_PLANET;
    if (!IsRayDescValid(r))
        return b;

    RayQuery<RAY_FLAG_NONE, RAYQUERY_FLAG_ALLOW_OPACITY_MICROMAPS> q;
    q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, r);
    [loop]
    for (uint it = 0u; q.Proceed() && it < 32u; ++it)
    {
        if (q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE) continue;
        const uint ci = q.CandidateInstanceID();
        const uint cp = FlatPrimID(ci, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
        const uint cm = GetMatIDFast(ci, cp);
        if (LoadIsOceanMaterial(cm)) continue;
        if (LoadIsThinGlass(cm) || LoadKd_w(cm) < 1.0f - EPSILON ||
            AlphaCandidateOccludes(ci, cp, q.CandidateTriangleBarycentrics()))
            q.CommitNonOpaqueTriangleHit();
    }
    if (q.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return b;

    const uint  hi   = q.CommittedInstanceID();
    const uint  hp   = FlatPrimID(hi, q.CommittedGeometryIndex(), q.CommittedPrimitiveIndex());
    const uint  hm   = GetMatIDFast(hi, hp);
    const float dist = q.CommittedRayT();

    HitInfo bh = EvalSurfaceState(hi, hp, q.CommittedTriangleBarycentrics(), r.Origin,
                                  PixelConeAngle() * (length(sv.x - camPos) + dist));
    float3 bKd; float bPr, bPm;
    RefetchMaterial(hm, bh, bKd, bPr, bPm);

    float3 sigmaA, sigmaS; float phaseG;
    OceanMediumCoefficients(sigmaA, sigmaS, phaseG);
    b.albedo = bKd;
    b.weight = transmitted * OceanMediumTransmittance(sigmaA + sigmaS, dist);
    b.hit    = true;
    return b;
}

// Mean albedo of the dithered whitecap steps, so the guide holds still.
float3 OceanGuideAlbedo(float foam, float bubbles)
{
    const uint base = OceanParams().materialBase;
    const float fx = saturate(foam) * (OCEAN_FOAM_STEPS - 1), fy = saturate(bubbles) * (OCEAN_BUBBLE_STEPS - 1);
    const uint f0 = min((uint)fx, (uint)OCEAN_FOAM_STEPS - 1u), b0 = min((uint)fy, (uint)OCEAN_BUBBLE_STEPS - 1u);
    const uint f1 = min(f0 + 1u, (uint)OCEAN_FOAM_STEPS - 1u), b1 = min(b0 + 1u, (uint)OCEAN_BUBBLE_STEPS - 1u);
    const float tf = fx - float(f0), tb = fy - float(b0);
    float3 albedo = 0.0f;
    [unroll] for (uint k = 0u; k < 4u; ++k) {
        const uint slot = base + ((k & 1u) != 0u ? f1 : f0) + OCEAN_FOAM_STEPS * ((k & 2u) != 0u ? b1 : b0);
        const float w = ((k & 1u) != 0u ? tf : 1.0f - tf) * ((k & 2u) != 0u ? tb : 1.0f - tb);
        albedo += w * LoadKd_rgb(slot) * LoadKd_w(slot);
    }
    return albedo;
}

void WriteOceanGuides(uint2 px, uint pixelIdx, float2 dims, float3 camPos, out DlssGuides g)
{
    const uint instID = load_instID(g_sample_current, pixelIdx);
    const float3 pos = load_x1(g_sample_current, pixelIdx);
    const SurfaceVertex sv = BuildVertex(g_sample_current, pixelIdx, pos, camPos);
    g.depth = DLSS_GuideDepthFromWorldPos(sv.x);
    g.n = sv.n_s;
    g.roughness = sv.Pr;
    const float3 fresnel = FresnelDielectricTIR(sv.o, sv.n_s, sv.etai, sv.etat);
    g.specularAlbedo = lerp(fresnel, FresnelConductor(sv.Kd, sv.o, sv.n_s), sv.Pm);
    // Own diffuse plus the bed below; whitecap steps use their mean albedo.
    const float3 below = 1.0f - fresnel;
    const OceanBelow bed = ResolveOceanBelow(sv, below, camPos);
    float3 own = sv.Kd * LoadKd_w(sv.matID);
    if (sv.matID - OceanParams().materialBase < (uint)OCEAN_MATERIAL_LEVELS) {
        const float4 foam = gScratchPing[uint3(px, OCEAN_GUIDE_SLOT)];
        own = OceanGuideAlbedo(foam.x, foam.y);
    }
    g.diffuseAlbedo = float4(own * (1.0f - sv.Pm) * below + bed.albedo * bed.weight, 1.0f);
    g.mv = SurfaceMotionVector(px, dims, sv.x, instID);
    g.specMv = g.mv;
    g.specHitDist = 0.0f;
    if (SHADING_DEBUG_SLICES)
        gOutput[uint3(px, 5)] = g.diffuseAlbedo;
}

// Underwater: fade the hit's guides into the medium's by transmittance.
void BlendUnderwaterGuides(uint2 px, float2 dims, float3 camPos, uint primaryInst, uint pixelIdx, inout DlssGuides g)
{
    float3 sigmaA, sigmaS; float phaseG;
    OceanMediumCoefficients(sigmaA, sigmaS, phaseG);
    const float3 sigmaT = sigmaA + sigmaS;
    const float3 vdir = PixelViewDir(px, dims);
    const bool hasHit = primaryInst != 0xFFFFFFFFu;
    const float hitDist = hasHit ? length(load_x1(g_sample_current, pixelIdx) - camPos) : 1e30f;
    // Clearest channel, not luminance; fogDist is where it falls to ~5%.
    const float clearest = max(min(sigmaT.x, min(sigmaT.y, sigmaT.z)), 1e-4f);
    const float fogDist = 3.0f / clearest;
    const float t = hasHit ? exp(-clearest * hitDist) : 0.0f;

    const float3 color = saturate(sigmaS / max(max(sigmaS.x, max(sigmaS.y, sigmaS.z)), 1e-6f));
    g.diffuseAlbedo = float4(lerp(color, g.diffuseAlbedo.rgb, t), 1.0f);
    g.specularAlbedo *= t;
    g.roughness = lerp(1.0f, g.roughness, t);
    g.n = normalize(lerp(-vdir, hasHit ? g.n : -vdir, t));
    g.depth = DLSS_GuideDepthFromWorldPos(camPos + vdir * min(hitDist, fogDist));
    // Faded too; a hard switch draws an arc across the view.
    g.mv = lerp(SkyMotionVector(px, dims), hasHit ? g.mv : 0.0f, t);
    g.specMv = lerp(g.mv, hasHit ? g.specMv : 0.0f, t);
    g.specHitDist *= t;
}

void WriteLegacyGuides(uint2 px, uint pixelIdx, float2 dims, float3 camPos, out DlssGuides g)
{
    const uint   sInstID = load_instID(g_sample_current, pixelIdx);
    const float3 sPos    = load_x1(g_sample_current, pixelIdx);
    const SurfaceVertex sv = BuildVertex(g_sample_current, pixelIdx, sPos, camPos);

    const RRGuide rg = ResolveRRGuideThroughGlass(sv, sInstID, camPos);

    g.depth = DLSS_GuideDepthFromWorldPos(rg.x);

    const float3 specularAlbedo = EnvBRDFApprox2(sv.Kd, sv.Pr, sv.Pm, dot(sv.o, sv.n_s));
    const float  reflW          = saturate(Luma(specularAlbedo));

    g.n = sv.n_s;
    g.roughness = sv.Pr;
    g.diffuseAlbedo = float4(rg.Kd, 1.0f);
    if (SHADING_DEBUG_SLICES) {
    gOutput[uint3(px, 5)] = float4(rg.Kd, 1.0f);
    }

    const float2 mvPixels = SurfaceMotionVector(px, dims, rg.x, rg.instID);
    g.mv = mvPixels;
    g.specularAlbedo = specularAlbedo;

    const float4 reflData   = gScratchPing[uint3(px, 4)];
    const uint   reflInstID = asuint(reflData.w);
    g.specHitDist = (reflInstID != 0xFFFFFFFFu)
        ? min(length(reflData.xyz - sv.x), DLSS_SPEC_HIT_MAX)
        : DLSS_SPEC_HIT_MAX;

    float2 specMV = LoadIsThinGlass(sv.matID) ? SurfaceMotionVector(px, dims, sv.x, sInstID) : mvPixels;
    if (!IS_OCEAN_INSTANCE(sInstID) && reflW > 0.04f && sv.Pr < DLSS_SPEC_ROUGHNESS_THRESHOLD && reflInstID != 0xFFFFFFFFu)
    {
        const float2 prevRefl = GetLastFramePixelCoordinates_Unclamped(reflData.xyz, prevView, prevProjection, dims, reflInstID);
        const float2 curRefl  = GetCurrentFramePixelCoordinates_Unclamped(reflData.xyz, view, projection, dims, reflInstID);
        if (prevRefl.x > -1e8f && curRefl.x > -1e8f)
            specMV = prevRefl - curRefl;
    }
    g.specMv = specMV;
}

// Primary surface replacement; returns the bias hint.
float WritePsrGuides(uint2 px, uint pixelIdx, float2 dims, float3 camPos, out DlssGuides g)
{
    const uint   sInstID = load_instID(g_sample_current, pixelIdx);
    const float3 sPos    = load_x1(g_sample_current, pixelIdx);
    const SurfaceVertex sv = BuildVertex(g_sample_current, pixelIdx, sPos, camPos);
    const float NoV        = saturate(dot(sv.o, sv.n_s));
    const bool  thin       = LoadIsThinGlass(sv.matID);
    const bool  seeThrough = PsrSeeThroughMaterial(sv.matID);

    // Shares use Fresnel toward the camera, not the pick probability.
    float3 mirror;            // energy share of the mirror image
    float3 residual;          // near-delta energy the probe misses
    float3 baseT;             // energy share reaching the base surface
    float  mirrorRoughness;
    PsrChainEnd base = PsrChainSurface(sv.x, sv.n_s, sv.Kd, sv.Pr, sv.Pm, sInstID);
    if (seeThrough)
    {
        // Reflection is the mirror; thin glass passes straight, clear glass refracts.
        const float3 F = thin ? FresnelDielectric(sv.o, sv.n_s, sv.etai, sv.etat)
                              : FresnelDielectricTIR(sv.o, sv.n_s, sv.etai, sv.etat);
        float3 dir = -sv.o;
        const bool  passes = thin || RefractVector(sv.o, sv.n_s, sv.etai / sv.etat, dir);
        const float fade   = PsrRoughnessFade(sv.Pr);
        mirror          = F * fade;
        residual        = F * (1.0f - fade);
        mirrorRoughness = sv.Pr;
        baseT = passes ? (1.0f - F) * (thin ? LoadTf(sv.matID) : 1.0f) : 0.0f;
        // Base defaults to the pane itself (rough glass blur).
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
        // Coat over GGX over diffuse.
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

    // Specular albedo stays the primary surface's own, never the chain's.
    const float3 primarySpecular = mirror + residual;

    // Base content counts as diffuse, even rough specular behind glass.
    const bool   baseSky     = (base.flags & DLSS_PSR_FLAG_SKY) != 0u;
    const bool   baseEmitter = (base.flags & DLSS_PSR_FLAG_EMITTER) != 0u;
    // Sky chain ends carry white; use the sky's own colour.
    const float3 baseKd      = baseSky ? SkyGuideAlbedo(-sv.o) : base.Kd;
    const float3 baseAlbedo  = (seeThrough && !baseSky && !baseEmitter)
        ? baseKd * (1.0f - base.Pm) + EnvBRDFApprox2(base.Kd, base.Pr, base.Pm, saturate(dot(base.nVirtual, sv.o)))
        : baseKd * (1.0f - base.Pm);
    const float3 baseDiffuse = baseT * baseAlbedo;

    // Virtual surface at the reflection chain's end.
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
    // Sky mirror image: colour of the reflected direction.
    const float3 KdV = virtualSky ? SkyGuideAlbedo(reflect(-sv.o, sv.n_s)) : probe.Kd;
    const float  PrV = virtualSky ? 1.0f : probe.Pr;
    const float  PmV = virtualSky ? 0.0f : probe.Pm;
    const float3 xV  = virtualSky ? camPos - sv.o * cameraFar : reflData.xyz;
    const float3 diffV = KdV * (1.0f - PmV);

    // Mirror's Fresnel share, reduced by the surface's own roughness.
    const float mirrorShare  = Luma(mirror);
    const float fresnelShare = mirrorShare / max(mirrorShare + Luma(baseDiffuse + residual), 1e-4f);
    const float w            = fresnelShare * (1.0f - saturate(sv.Pr));

    const float3 diffuseAlbedo  = baseDiffuse + mirror * diffV * w;
    const float3 specularAlbedo = primarySpecular;

    // Glass: the surface behind; opaque: its own mirror lobe.
    const float baseRoughness = seeThrough ? base.Pr : mirrorRoughness;
    const float roughness     = lerp(baseRoughness, PrV, w);

    float3 n = lerp(base.nVirtual, nV, w);
    n = (dot(n, n) > 1e-6f) ? normalize(n) : (w > 0.5f ? nV : base.nVirtual);

    // Blended in encoded (reciprocal) depth.
    const float depth = lerp(DLSS_GuideDepthFromWorldPos(base.xVirtual), DLSS_GuideDepthFromWorldPos(xV), w);

    g.depth          = depth;
    g.n              = n;
    g.roughness      = roughness;
    g.diffuseAlbedo  = float4(diffuseAlbedo, 1.0f);
    g.specularAlbedo = specularAlbedo;
    if (SHADING_DEBUG_SLICES) {
    gOutput[uint3(px, 5)] = float4(diffuseAlbedo, 1.0f);
    }

    // One reflection is re-mirrored exactly; longer chains follow the end instance.
    const float2 baseMV    = PsrChainMotionVector(px, dims, base);
    float2 virtualMV = virtualSky ? SkyMotionVector(px, dims)
        : (probe.bounces <= 1u
            ? PsrVirtualMotionVector(reflData.xyz, reflInstID, sv.x, sv.n_s, sInstID, dims)
            : SurfaceMotionVector(px, dims, reflData.xyz, reflInstID));
    // Pane at the chain end: its reflection moves like the sky.
    if ((probe.flags & DLSS_PSR_FLAG_GLASS) != 0u)
        virtualMV = lerp(virtualMV, SkyMotionVector(px, dims), probe.paneF);

    // Opaque only; glass keeps the transmission target's motion.
    const float mvWeight = lerp(w > 0.5f ? 1.0f : 0.0f, w, DLSS_PSR_MV_MIX);
    const bool  mvBlend  = !seeThrough && (dbg_dlssLayer & DLSS_GUIDE_OPT_NO_MV_BLEND) == 0u;
    g.mv = mvBlend ? lerp(baseMV, virtualMV, mvWeight) : baseMV;

    const bool   mirrorLayer = mirrorRoughness < DLSS_SPEC_ROUGHNESS_THRESHOLD && Luma(mirror + residual) > 0.02f;
    const float2 surfaceMV   = seeThrough ? SurfaceMotionVector(px, dims, sv.x, sInstID) : baseMV;
    g.specMv      = mirrorLayer ? virtualMV : surfaceMV;
    g.specHitDist = (asuint(reflFirst.w) != 0xFFFFFFFFu)
        ? min(length(reflFirst.xyz - sv.x), DLSS_SPEC_HIT_MAX) : DLSS_SPEC_HIT_MAX;

    // Emitters behind delta chains count as direct, by Fresnel share.
    float bias = 0.0f;
    if (virtualEmitter && mirrorRoughness < SMOOTH_SPECULAR_THRESHOLD && (probe.flags & DLSS_PSR_FLAG_DELTA) != 0u)
        bias += fresnelShare;
    if (seeThrough && baseEmitter && sv.Pr < SMOOTH_SPECULAR_THRESHOLD && (base.flags & DLSS_PSR_FLAG_DELTA) != 0u)
        bias += 1.0f - fresnelShare;
    return saturate(bias);
}


[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= IMG_W || DTid.y >= IMG_H) return;

    const float3 camPosWorld = mul(viewI, float4(0, 0, 0, 1)).xyz;
    // Needed by SkyGuideAlbedo.
    SetSkyObserver(camPosWorld + sceneOriginWorld);

    float3 output_primary  = gScratchPing[uint3(DTid.xy, 1)].rgb;
    float3 output_indirect = gScratchPing[uint3(DTid.xy, 2)].rgb;

    const float3 dbgRawPrimary  = output_primary;
    const float3 dbgRawIndirect = output_indirect;

    const float3 atmosphereL = gScratchPing[uint3(DTid.xy, 10)].rgb;
    const float3 atmosphereTr = gScratchPing[uint3(DTid.xy, 11)].rgb;

    float3 accumulation = (output_primary + output_indirect) * atmosphereTr + atmosphereL;

    // Underwater paths already carry their transport; no atmosphere.
    const uint waterPixel = MapPixelID(uint2(IMG_W,IMG_H),DTid.xy);
    if ((load_flagsWord(g_sample_current,waterPixel) & SD_FLAG_CAMERA_WATER) != 0u)
        accumulation = output_primary+output_indirect;
    if (SHADING_DEBUG_SLICES) {
    gScratchPing[uint3(DTid.xy, 1)] = float4(accumulation, 0);
    }

    if (ATM_DEBUG_RING == 2u) {
    {
        float dr = 1.0f - exp(-max(Luma(dbgRawIndirect), 0.0f) * 3.0f);
        float dg = 0.0f;
        float db = 1.0f - exp(-max(Luma(dbgRawPrimary), 0.0f) * 3.0f);
        accumulation = float3(dr, dg, db);
    }
    }

    if (SHADING_DEBUG_SLICES) {
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
    }

    float2 dims = float2(IMG_W, IMG_H);
    uint   pixelIdx  = MapPixelID(dims, DTid.xy);

    bool  isEmitterSurface = false;
    float psrBias = 0.0f;
    DlssGuides g;
    float3 input;

    const uint primaryInst = load_instID(g_sample_current, pixelIdx);

    bool isEmissiveOrSky = load_isEmitter(g_sample_current, pixelIdx);
    if (isEmissiveOrSky)
    {

        uint emInstID = load_instID(g_sample_current, pixelIdx);
        bool hasPosition = (emInstID != 0xFFFFFFFFu);

        if (hasPosition)
        {

            float3 emPos  = load_x1(g_sample_current, pixelIdx);
            g.depth = DLSS_GuideDepthFromWorldPos(emPos);
            g.mv = SurfaceMotionVector(DTid.xy, dims, emPos, emInstID);
            isEmitterSurface = true;

            g.n = load_n1_s_with_instID(g_sample_current, pixelIdx, emInstID);
        }
        else
        {
            g.depth = 0.0f;
            g.n = float3(0.0f, 0.0f, 0.0f);
            g.mv = SkyMotionVector(DTid.xy, dims);
        }
        g.roughness = 1.0f;

        g.specularAlbedo = float3(0.0f, 0.0f, 0.0f);

        const float3 emitterAlbedo = saturate(ScrubNonFiniteIn(dbgRawPrimary));
        g.diffuseAlbedo = float4(hasPosition ? emitterAlbedo : SkyGuideAlbedo(PixelViewDir(DTid.xy, dims)), 0.0f);

        g.specHitDist = hasPosition ? 0.0f : DLSS_SPEC_HIT_MAX;
        g.specMv = float2(0.0f, 0.0f);

        float3 emitterRadiance = (CLAMP_EMITTERS_MODE && hasPosition)
                                    ? ClampEmitterLum(accumulation)
                                    : accumulation;
        input = DlssEncode(emitterRadiance);
        if (hasPosition) {
            const float lum = dot(input, float3(0.2126f, 0.7152f, 0.0722f));
            if (lum > DLSS_PT_INPUT_LUMA_CAP)
                input *= DLSS_PT_INPUT_LUMA_CAP / lum;
        }
    if (SHADING_DEBUG_SLICES) {
        gOutput[uint3(DTid.xy, 5)] = float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
    }
    else{
        // Water deforms: keeps its own guides, no PSR.
        if (IS_OCEAN_INSTANCE(primaryInst))
            WriteOceanGuides(DTid.xy, pixelIdx, dims, camPosWorld, g);
        else if ((dbg_dlssLayer & DLSS_GUIDE_OPT_NO_PSR) != 0u)
            WriteLegacyGuides(DTid.xy, pixelIdx, dims, camPosWorld, g);
        else
            psrBias = WritePsrGuides(DTid.xy, pixelIdx, dims, camPosWorld, g);

    if (ATM_DEBUG_RING == 4u) {

        input = g.n * 0.5f + 0.5f;
    } else {
        input = DlssEncode(accumulation);
    }

    }

    if ((load_flagsWord(g_sample_current, pixelIdx) & SD_FLAG_CAMERA_WATER) != 0u)
        BlendUnderwaterGuides(DTid.xy, dims, camPosWorld, primaryInst, pixelIdx, g);

    float normalW = g.roughness;
    if ((rs_flags & RS_FLAG_GUIDE_OFF_ANY) != 0u)
    {
        if ((rs_flags & RS_FLAG_GUIDE_OFF_DEPTH)   != 0u) g.depth = 0.0f;
        if ((rs_flags & RS_FLAG_GUIDE_OFF_MV)      != 0u) g.mv = float2(0.0f, 0.0f);
        if ((rs_flags & RS_FLAG_GUIDE_OFF_NORMALS) != 0u) g.n = float3(0.0f, 0.0f, 0.0f);
        if ((rs_flags & RS_FLAG_GUIDE_OFF_ROUGH)   != 0u) normalW = 1.0f;
        if ((rs_flags & RS_FLAG_GUIDE_OFF_ALBEDO)  != 0u) g.diffuseAlbedo  = float4(1.0f, 1.0f, 1.0f, 1.0f);
        if ((rs_flags & RS_FLAG_GUIDE_OFF_SPECALB) != 0u) g.specularAlbedo = float3(0.0f, 0.0f, 0.0f);
        if ((rs_flags & RS_FLAG_GUIDE_OFF_SPECMV)  != 0u) g.specMv = float2(0.0f, 0.0f);
    }

    g_dlssInput[DTid.xy]          = float4(input, 1.0f);
    g_dlssDepth[DTid.xy]          = g.depth;
    g_dlssMVec[DTid.xy]           = g.mv;
    g_dlssNormals[DTid.xy]        = float4(g.n, normalW);
    g_dlssRoughness[DTid.xy]      = g.roughness;
    g_dlssDiffuseAlbedo[DTid.xy]  = g.diffuseAlbedo;
    g_dlssSpecularAlbedo[DTid.xy] = float4(g.specularAlbedo, 0.0f);
    g_dlssSpecMVec[DTid.xy]       = g.specMv;
    g_dlssSpecHitDist[DTid.xy]    = g.specHitDist;
    g_dlssBiasHint[DTid.xy]       = isEmitterSurface ? 1.0f : psrBias;
    // Water has its own (glitter changes every frame); else a roughness ramp.
    g_dlssResponsivity[DTid.xy]   = IS_OCEAN_INSTANCE(primaryInst)
                                        ? dlssWaterResponsivity
                                        : lerp(dlssResponsivityMirror, dlssResponsivityRough,
                                               saturate(g.roughness));
    g_dlssTransparency[DTid.xy]   = float4(0.0f, 0.0f, 0.0f, 0.0f);

    {
        const float3 sCol  = input;
        const float  sDep  = g.depth;
        const float2 sMV   = g.mv;
        const float4 sNR   = float4(g.n, normalW);
        const float2 sSMV  = g.specMv;
        const float3 sAlb  = g.diffuseAlbedo.rgb;
        const float3 sSAlb = g.specularAlbedo;

        const float sLum   = 0.2126f * sCol.x + 0.7152f * sCol.y + 0.0722f * sCol.z;
        const float sMvMag = max(abs(sMV.x),  abs(sMV.y));
        const float sSmMag = max(abs(sSMV.x), abs(sSMV.y));

        uint bad = 0u;
        if (any(isnan(sCol))      || any(isinf(sCol)))        bad |= 0x001u;
        if (isnan(sDep) || isinf(sDep) || sDep < 0.0f || sDep > 1.0f)
                                                              bad |= 0x002u;
        if (any(isnan(sMV))       || any(isinf(sMV)))         bad |= 0x004u;
        if (any(isnan(sNR))       || any(isinf(sNR)))         bad |= 0x008u;
        if (sNR.w < 0.0f || sNR.w > 1.0f)                     bad |= 0x010u;
        if (any(isnan(sSMV))      || any(isinf(sSMV)))        bad |= 0x020u;
        if (any(isnan(sAlb))      || any(isinf(sAlb)))        bad |= 0x040u;
        if (any(isnan(sSAlb))     || any(isinf(sSAlb)))       bad |= 0x080u;
        if (sMvMag > 256.0f)                                  bad |= 0x100u;
        if (sSmMag > 256.0f)                                  bad |= 0x200u;

        if (bad != 0u)
            gAutoExpose.InterlockedCompareStore(SENT_OFFS_FIRSTBAD, 0u,
                                                ((DTid.y + 1u) << 16) | (DTid.x + 1u));

        const bool  nearCap = isEmitterSurface &&
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
