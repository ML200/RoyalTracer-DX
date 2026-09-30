#include "Includes_v8.hlsli"
#include "OceanVolume.hlsli"
#include "PtVertex_v8.hlsli"

// One path sample; leaves the deferred record for the light and material passes.
[shader("raygeneration")]
void Pass_pt_trace_v8()
{
    const uint2 pixel   = DispatchRaysIndex().xy;
    const uint2 imgSize = uint2(IMG_W, IMG_H);
    if (any(pixel >= imgSize)) return;
    const uint pixelIdx = MapPixelID(imgSize, pixel);
    const uint s = PT_SAMPLE_INDEX;

    const uint cameraFlags = load_flagsWord(g_sample_current,pixelIdx);
    const bool cameraWater = (cameraFlags & SD_FLAG_CAMERA_WATER) != 0u;
    if ((cameraFlags & SD_FLAG_NOBOUNCE) != 0u && !cameraWater)
    {
        DvStoreInfo(pixelIdx, 0u);
        return;
    }
    if (s == 0u) gScratchPing[uint3(pixel, 2)] = 0.0f;

    const uint maxBounces = pt_maxBounces;

    // Absolute until the deferred vertex, then relative to its scatter (relL).
    float3 throughput = float3(1, 1, 1);
    float3 gathered   = 0.0f;
    float  prev_pdf   = 1.0f;
    float  pathSpread = 0.0f;
    uint   pathCone   = 0u;     // lobe cone for the cache test (PtConePack)
    float3 liteL      = 0.0f;
    float3 liteSuffix = 0.0f;
    uint   ps = PtPsInit(s);

    // Deferred vertex state.
    bool   pending   = false;
    bool   immediate = false;
    uint   info      = 0u;
    DvMiss missEnd = (DvMiss)0;

    // The first iteration shades the camera record, untraced.
    float3 pos = load_x1(g_sample_current, pixelIdx);   // the vertex the next ray leaves
    const float3 toPrimary = pos - InitOrigin();
    float  pathDist = length(toPrimary);   // for the texture footprint
    float3 rayDir   = toPrimary / max(pathDist, 1e-6f);
    RayDesc ray;
    ray.Origin    = pos;
    ray.Direction = rayDir;
    ray.TMin      = 0.0f;
    ray.TMax      = 0.0f;
    uint inFlags = PvInputFlags(ps, false, false, false);
    bool cameraRecordPending = !cameraWater;
    if (cameraWater) {
        uint cameraSeed = initRandomData(pixel,uint2(8,4),time,1u);
        InitCameraRayDoF(pixel,imgSize,cameraSeed,ray.Origin,rayDir);
        ray.Direction = rayDir; ray.TMin = 0.00001f; ray.TMax = RAY_TMAX_PLANET;
        pos = ray.Origin; pathDist = 0.0f;
        inFlags |= PV_IN_WATER_MEDIUM;
        if (LITE_ENABLED && s == 0u) LiteMarkEmpty(pixelIdx);
    }
    uint prevNPk = PackNormal(float3(0.0f, 0.0f, 1.0f));   // shading normal of the vertex the ray left

    [loop]
    for (;;)
    {
        const uint depth = PtPsDepth(ps);
        OceanPathHit hit = (OceanPathHit)0;
        uint reorderHint = 0x100u;
        if (cameraRecordPending)
            reorderHint = load_instID(g_sample_current, pixelIdx) & 0xffu;
        else
        {
            hit = OceanTracePathHitInMedium(ray, (inFlags & PV_IN_WATER_MEDIUM) != 0u);
            if ((inFlags & PV_IN_WATER_MEDIUM) != 0u) {
                // Water in-scatter; only the first segment samples the sun and lights.
                uint sVolume = RcBounceSeed(PtPathSeed(pixel, s), depth, 0x57415452u);
                const float segment = OceanBoundaryDistance(ray.Origin, rayDir, hit.distance);
                const bool sampled = (inFlags & PV_IN_WATER_SCATTERED) == 0u && depth <= (cameraWater ? 1u : 2u);
                const float3 inScatter = OceanSegmentInScatter(ray.Origin, rayDir, segment, sampled, sVolume);
                gathered += throughput * inScatter;
                if (depth >= 2u && (ps & PT_PS_LITE_VERTEX) != 0u) liteL += liteSuffix * inScatter;
                float3 sigmaA, sigmaS; float phaseG;
                OceanMediumCoefficients(sigmaA, sigmaS, phaseG);
                const float3 segmentT = OceanMediumTransmittance(sigmaA + sigmaS, segment);
                throughput *= segmentT;
                if ((ps & PT_PS_LITE_VERTEX) != 0u) liteSuffix *= segmentT;
                // Deferred emitter replay can't attenuate; resolve in place.
                immediate = false;
                inFlags &= ~PV_IN_IMMEDIATE;
                inFlags |= PV_IN_WATER_SCATTERED;
                if (!any(throughput > 0.0f)) break;
            }
            if (hit.hit) reorderHint = hit.instance & 0xffu;
        }
        // Keep outside the trace branch: reordering inside it hung the device.
#if PT_SER_REORDER
        dx::MaybeReorderThread(reorderHint | (pending ? 0x200u : 0u), 10u);
#endif

        // One PtVertexShade call site, so it inlines once.
        PtVertexIO io;
        io.flags = inFlags; io.pdf = prev_pdf; io.spread = pathSpread; io.dist = pathDist; io.cone = pathCone;
        io.dirPk = 0u; io.nPk = 0u; io.pos = pos; io.color = 0.0f; io.auxPk = 0u; io.nee = 0.0f; io.neeLite = 0.0f;
        HitContext ctx = (HitContext)0;
        float3 geoN      = 0.0f;
        bool   flipIOR   = false;
        uint   presetOut = 0u;
        bool   shade     = false;
        if (cameraRecordPending)
        {
            PtPrimaryContext(pixel, pixelIdx, ctx, geoN, flipIOR);
            shade = true;
        }
        else if (hit.hit)
        {
            shade = PtHitContext(io, hit.instance, hit.geometry, hit.primitive, hit.barycentrics, hit.distance,
                rayDir, pos, UnpackNormal(prevNPk), pixelIdx, ctx, geoN, flipIOR, presetOut);
        }
        else
            PtShadeMiss(io, pos, rayDir);
        if (shade) PtVertexShade(io, ctx, geoN, rayDir, flipIOR, pixel, pixelIdx, presetOut);
        if (depth == 1u)
        {
            if (LITE_ENABLED && s == 0u && (io.flags & PV_LITE_NEE) == 0u) LiteMarkEmpty(pixelIdx);
            ps = PtPsWith(ps, PT_PS_LITE_VERTEX, (io.flags & PV_PRIMARY_LITE) != 0u);
        }
        pos = io.pos;
        cameraRecordPending = false;

        const uint result = io.flags & PV_RESULT_MASK;
        const bool liteVertex = (ps & PT_PS_LITE_VERTEX) != 0u;
        const bool litePath   = depth >= 2u && liteVertex;

        if (result == PV_MISS)
        {
            const float3 sky = io.color;
            const float3 sunRad = PvUnpackRadiance(io.auxPk);
            const float  sunMis = io.pdf;
            const float3 envL = sky + sunRad * sunMis;
            gathered += throughput * envL;
            if (pending)
            {
                if (litePath) liteL += liteSuffix * envL;
                missEnd.dir = rayDir;
                missEnd.skySun = sky + sunRad;
                missEnd.candMis = sunMis;
                info |= (DV_END_MISS << DV_END_SHIFT) | ((immediate && (info & DV_LITE_GEN) != 0u) ? DV_LITE_MISS : 0u);
            }
            break;
        }
        if (result == PV_EMITTER)
        {
            const float3 emission = PvUnpackRadiance(io.auxPk);
            gathered += throughput * emission;
            if (litePath) liteL += liteSuffix * emission;
            break;
        }
        if (result == PV_EMITTER_DEFERRED)
        {
            info |= DV_END_EMITTER << DV_END_SHIFT;
            break;
        }

        // Park first: a cache hit here is the suffix's first radiance.
        if ((io.flags & PV_LITE_PARKED) != 0u)
        {
            liteSuffix = 1.0f;
            liteL = 0.0f;
            ps |= PT_PS_LITE_X2;
        }
        if ((io.flags & PV_CACHE_HIT) != 0u)
        {
            const float3 cached = PvUnpackRadiance(io.auxPk);
            gathered += throughput * cached;
            if (litePath)
            {
                liteL += liteSuffix * cached;
                liteSuffix = 0.0f;
                if (!any(throughput > 0.0f)) break;
            }
        }
        if ((io.flags & PV_NEE) != 0u)
        {
            gathered += throughput * io.nee;
            if (litePath) liteL += liteSuffix * io.neeLite;
        }
        // The subsurface exit counts as one more depth.
        if ((io.flags & PV_SSS_WALKED) != 0u)
        {
            ps |= PT_PS_SSS;
            ps += 1u << PT_PS_DEPTH_SHIFT;
            if (PtPsGuideDepth(ps) < 15u) ps += 1u << PT_PS_GUIDE_SHIFT;
        }
        const uint kind = (io.flags >> PV_KIND_SHIFT) & DV_KIND_MASK;
        const bool captured = (io.flags & PV_CAPTURED) != 0u;
        if (captured)
        {
            // Flush to the pixel; gather relative to the deferred scatter from here.
            PtAddRadiance(pixel, gathered);
            gathered = 0.0f;
            pending = true;
            info = DV_VALID | kind | (((io.flags >> PV_STRATEGY_SHIFT) & 3u) << DV_STRATEGY_SHIFT) |
                ((io.flags & PV_MIS_NONE) != 0u ? DV_MIS_NONE : 0u) |
                ((PtPsDepth(ps) == 1u && (ps & PT_PS_LITE_VERTEX) != 0u) ? DV_LITE_GEN : 0u);
            DvStorePathHead(pixelIdx, throughput * io.color, ps);
            throughput = 1.0f;
            immediate = kind == DV_KIND_BSDF;
        }
        else if (result == PV_CONTINUE)
        {
            throughput *= io.color;
            if (litePath)
                liteSuffix *= (io.flags & PV_AUX_BROAD) != 0u ? UnpackRGB9E5(io.auxPk) : io.color;
            immediate = (io.flags & PV_PASS_THROUGH) != 0u ? immediate : false;
        }
        if ((io.flags & PV_DIFF_INC) != 0u) ps += 1u << PT_PS_DIFF_SHIFT;
        ps = PtPsWith(ps, PT_PS_SPREAD, (io.flags & PV_SPREAD) != 0u);
        if ((io.flags & PV_SPREAD) != 0u && PtPsGuideDepth(ps) < 15u) ps += 1u << PT_PS_GUIDE_SHIFT;
        ps = PtPsWith(ps, PT_PS_MIS_NONE, (io.flags & PV_MIS_NONE) != 0u);
        prev_pdf   = io.pdf;
        pathSpread = io.spread;
        pathCone   = io.cone;
        pathDist   = io.dist;
        if (result == PV_TERMINATE) break;

        // Russian roulette; the material pass replays it for deferred scatters.
        if (!(captured && kind == DV_KIND_BSDF) && PtPsDepth(ps) >= (uint)pt_rrStartDepth)
        {
            uint sRr = RcBounceSeed(PtPathSeed(pixel, s), PtPsDepth(ps), RC_STREAM_RR);
            const float survivalProb = max(min(1.0f, Luma(throughput)), 0.05f);
            if (RandomFloatSingle(sRr) >= survivalProb) break;
            throughput /= survivalProb;
            if (litePath) liteSuffix *= rcp(survivalProb);
        }

        rayDir = UnpackNormal(io.dirPk);
        const float3 vertexN = UnpackNormal(io.nPk);
        prevNPk = io.nPk;
        ray.Origin    = offset_ray(pos, (dot(rayDir, vertexN) >= 0.0f) ? vertexN : -vertexN);
        ray.Direction = rayDir;
        ray.TMin      = 0.00001f;
        ray.TMax      = RAY_TMAX_PLANET;
        if (!IsRayDescValid(ray))
        {
            if (pending && immediate) info = (info & ~DV_KIND_MASK) | DV_KIND_NONE;
            break;
        }
        const uint nextDepth = PtPsDepth(ps) + 1u;
        if (nextDepth > maxBounces) break;
        ps += 1u << PT_PS_DEPTH_SHIFT;
        inFlags = PvInputFlags(ps, pending, immediate, nextDepth >= maxBounces);
        inFlags |= (io.flags & PV_WATER_DIRECT) != 0u ? PV_IN_WATER_DIRECT : 0u;
        inFlags |= (io.flags & PV_WATER_MEDIUM) != 0u ? PV_IN_WATER_MEDIUM : 0u;
        inFlags |= (io.flags & PV_WATER_SCATTERED) != 0u ? PV_IN_WATER_SCATTERED : 0u;
        inFlags |= (io.flags & PV_SUN_OWNED) != 0u ? PV_IN_SUN_OWNED : 0u;
    }

    if (!pending)
    {
        PtAddRadiance(pixel, gathered);
        DvStoreInfo(pixelIdx, 0u);
        return;
    }
    if ((ps & PT_PS_LITE_X2) != 0u)
    {
        info |= DV_LITE_X2;
        DvStoreLiteL(pixelIdx, liteL);
    }
    DvStoreInfo(pixelIdx, info);
    DvStorePathRelL(pixelIdx, gathered);
    if (((info >> DV_END_SHIFT) & DV_END_MASK) == DV_END_MISS) DvStoreMiss(pixelIdx, missEnd);
}
