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

    uint  biasInstID;
    float2 biasMV = float2(0, 0);
    bool  isEmitterSurface = false;

    bool isEmissiveOrSky = load_isEmitter(g_sample_current, pixelIdx);
    if (isEmissiveOrSky)
    {

        uint emInstID = load_instID(g_sample_current, pixelIdx);
        bool hasPosition = (emInstID != 0xFFFFFFFFu);

        if (hasPosition)
        {

            float3 emPos  = load_x1(g_sample_current, pixelIdx);
            g_dlssDepth[DTid.xy] = DLSS_GuideDepthFromWorldPos(emPos);

            float2 curPix = DTid.xy;

            float2 prevPix = GetLastFramePixelCoordinates_Unclamped(
                emPos, prevView, prevProjection, dims, emInstID);
            float2 curPinholePix = GetCurrentFramePixelCoordinates_Unclamped(
                emPos, view, projection, dims, emInstID);
            bool validPrev = (prevPix.x > -1e8f) && (curPinholePix.x > -1e8f);
            float2 emMV = validPrev ? (prevPix - curPinholePix) : float2(0, 0);
            g_dlssMVec[curPix] = emMV;
            biasMV = emMV;
            isEmitterSurface = true;

            const float3 emNormal = load_n1_s_with_instID(g_sample_current, pixelIdx, emInstID);

            g_dlssNormals[DTid.xy] = float4(emNormal, 1.0f);
        }
        else
        {
            g_dlssDepth[DTid.xy] = 0.0f;
            g_dlssNormals[DTid.xy] = float4(0.0f, 0.0f, 0.0f, 1.0f);

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

                    skyMV = prevPix - float2(DTid.xy);
                }
            }
            g_dlssMVec[DTid.xy] = skyMV;
            biasMV = skyMV;

        }

        biasInstID = emInstID;
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
        uint sInstID = load_instID(g_sample_current, pixelIdx);
        float3 sPos  = load_x1(g_sample_current, pixelIdx);
        SurfaceVertex sv = BuildVertex(g_sample_current, pixelIdx, sPos, camPosWorld);

        const RRGuide rg = ResolveRRGuideThroughGlass(sv, sInstID, camPosWorld);

        g_dlssDepth[DTid.xy] = DLSS_GuideDepthFromWorldPos(rg.x);

        float3 specularAlbedo = EnvBRDFApprox2(sv.Kd, sv.Pr, sv.Pm, dot(sv.o, sv.n_s));
        float  reflW          = saturate(Luma(specularAlbedo));

        g_dlssNormals[DTid.xy] = float4(sv.n_s, sv.Pr);

        g_dlssDiffuseAlbedo[DTid.xy] = float4(rg.Kd, 1.0f);
        g_dlssRoughness[DTid.xy] = sv.Pr;

#if SHADING_DEBUG_SLICES
        gOutput[uint3(DTid.xy, 5)] = float4(rg.Kd, 1.0f);
#endif

        float2 curPix = DTid.xy;

        float2 prevPix       = GetLastFramePixelCoordinates_Unclamped(rg.x, prevView, prevProjection, dims, rg.instID);
        float2 curPinholePix = GetCurrentFramePixelCoordinates_Unclamped(rg.x, view, projection, dims, rg.instID);

        bool validPrev = (prevPix.x > -1e8f) && (curPinholePix.x > -1e8f);

        float2 mvPixels = validPrev ? (prevPix - curPinholePix) : float2(0.0, 0.0);

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

                    skyMV = prevPixSky - float2(DTid.xy);
                }
            }
            mvPixels = skyMV;
        }

        g_dlssMVec[curPix] = mvPixels;

        biasInstID = sInstID;
        biasMV = mvPixels;

        g_dlssSpecularAlbedo[DTid.xy] = float4(specularAlbedo, 0.0f);

        const float4 reflData   = gScratchPing[uint3(DTid.xy, 4)];
        const uint   reflInstID = asuint(reflData.w);
        g_dlssSpecHitDist[DTid.xy] = (reflInstID != 0xFFFFFFFFu)
            ? min(length(reflData.xyz - sv.x), DLSS_SPEC_HIT_MAX)
            : DLSS_SPEC_HIT_MAX;

        float2 surfaceMV = mvPixels;
        if (LoadIsThinGlass(sv.matID))
        {
            float2 gPrev = GetLastFramePixelCoordinates_Unclamped(sv.x, prevView, prevProjection, dims, sInstID);
            float2 gCur  = GetCurrentFramePixelCoordinates_Unclamped(sv.x, view, projection, dims, sInstID);
            if (gPrev.x > -1e8f && gCur.x > -1e8f)
                surfaceMV = gPrev - gCur;
        }

        float2 specMV = surfaceMV;
        if (reflW > 0.04f && sv.Pr < DLSS_SPEC_ROUGHNESS_THRESHOLD)
        {

            if (reflInstID != 0xFFFFFFFFu)
            {

                float2 prevRefl = GetLastFramePixelCoordinates_Unclamped(
                    reflData.xyz, prevView, prevProjection, dims, reflInstID);
                float2 curRefl  = GetCurrentFramePixelCoordinates_Unclamped(
                    reflData.xyz, view, projection, dims, reflInstID);
                if (prevRefl.x > -1e8f && curRefl.x > -1e8f)
                    specMV = prevRefl - curRefl;
            }
        }
        g_dlssSpecMVec[DTid.xy] = specMV;

#if ATM_DEBUG_RING == 4

        g_dlssInput[DTid.xy] = float4(sv.n_s * 0.5f + 0.5f, 1.0f);
#else
        g_dlssInput[DTid.xy] = float4(DlssEncode(accumulation), 1.0f);
#endif

    }

    ApplyCumulusGuides(DTid.xy,pixelIdx,camPosWorld);
    float cloudOpacity=gScratchPing[uint3(DTid.xy,CUMULUS_NORMAL_SLOT)].w;
    g_dlssBiasHint[DTid.xy] = isEmitterSurface ? (1.0f-cloudOpacity) : 0.0f;

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
