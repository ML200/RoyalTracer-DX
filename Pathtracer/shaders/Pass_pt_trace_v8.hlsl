#include "Includes_v8.hlsli"
#include "PtVertex_v8.hlsli"
#include "OceanVolume.hlsli"

// Drive one sample of the pixel path. Every vertex is shaded after the reorder, the primary one
// straight from the camera record without a trace; the path loop keeps the path state,
// applies the compact results and finalizes the deferred record for the light and material passes.
// Whatever the loop carries is saved and restored around each trace and reorder, so the part of
// the deferred record that is known at the capture is written there and not carried to the end.
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

    // The throughput is absolute until the deferred vertex and relative to its scatter afterwards;
    // the radiance gathered with it is the pixel's until then and the record's relL afterwards.
    float3 throughput = float3(1, 1, 1);
    float3 gathered   = 0.0f;
    float  prev_pdf   = 1.0f;
    float  pathSpread = 0.0f;
    uint   pathCone   = 0u;     // cone of the path's lobes for the cache test (PtConePack)
    float3 liteL      = 0.0f;
    float3 liteSuffix = 0.0f;
    uint   ps = PtPsInit(s);

    // Deferred vertex state.
    bool   pending   = false;
    bool   immediate = false;
    uint   info      = 0u;
    DvMiss missEnd = (DvMiss)0;

    // The camera pass resolved the primary vertex, so the first iteration shades it from the camera
    // record without a trace.
    float3 pos = load_x1(g_sample_current, pixelIdx);   // the vertex the next ray leaves
    const float3 toPrimary = pos - InitOrigin();
    float  pathDist = length(toPrimary);   // path length: with the pixel cone it sizes the texture footprint
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
        // ---- trace, reorder ----
        const uint depth = PtPsDepth(ps);
        OceanPathHit hit = (OceanPathHit)0;
        uint reorderHint = 0x100u;
        if (cameraRecordPending)
            reorderHint = load_instID(g_sample_current, pixelIdx) & 0xffu;
        else
        {
            hit = OceanTracePathHit(ray);
            if ((inFlags & PV_IN_WATER_MEDIUM) != 0u) {
                uint sVolume = RcBounceSeed(PtPathSeed(pixel, s), depth, 0x57415452u);
                const OceanFlight flight = OceanSampleWaterSegment(ray.Origin,rayDir,
                    hit.distance,
                    (inFlags & PV_IN_WATER_SCATTERED) != 0u,sVolume);
                throughput *= flight.weight;
                if ((ps & PT_PS_LITE_VERTEX) != 0u) liteSuffix *= flight.weight;
                // A deferred emitter replay has no slot for segment attenuation.
                // Resolve this endpoint in place using the now-attenuated throughput.
                immediate = false;
                inFlags &= ~PV_IN_IMMEDIATE;
                if (!any(throughput > 0.0f)) break;
                if (flight.scattered) {
                    // Replace the endpoint with the volume vertex, then trace its
                    // phase-sampled continuation. Only object/surface bounces consume
                    // the surface-bounce budget; the scattered flag bounds this loop.
                    pos = ray.Origin+rayDir*flight.distance;
                    pathDist += flight.distance;
                    rayDir = OceanSampleScatterDirection(rayDir,sVolume);
                    ray.Origin = pos; ray.Direction = rayDir;
                    ray.TMin = 0.00001f; ray.TMax = RAY_TMAX_PLANET;
                    ps |= PT_PS_MIS_NONE;
                    inFlags = (inFlags | PV_IN_WATER_SCATTERED | PV_IN_MIS_NONE) & ~PV_IN_WATER_DIRECT;
                    // There is no NEE partner at a volume vertex. Preserve emission
                    // and sky reached by this new path, and don't apply surface MIS.
                    prev_pdf = 1.0f;
                    continue;
                }
            }
            if (hit.hit) reorderHint = hit.instance & 0xffu;
        }
        // The one reorder point, on the path of every iteration, the primary one sorted by the
        // instance of the camera record: with the reorder inside the trace branch (skipped on the
        // primary iteration) the device hung at random. Only a completed surface/miss endpoint
        // reaches it; the volume continuation above carries ordinary data, never a live HitObject.
#if PT_SER_REORDER
        dx::MaybeReorderThread(reorderHint, 9u);
#endif

        // ---- shade: build the context of the vertex, then the one shading call ----
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
            if (LITE_ENABLED && s == 0u && (io.flags & PV_PRIMARY_LITE) == 0u) LiteMarkEmpty(pixelIdx);
            ps = PtPsWith(ps, PT_PS_LITE_VERTEX, (io.flags & PV_PRIMARY_LITE) != 0u);
        }
        pos = io.pos;
        cameraRecordPending = false;

        // ---- apply the vertex result ----
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

        // The park comes first: the reuse suffix starts at this vertex, so a cache hit at the
        // parked point is the first thing it collects (the hit shader reports both together).
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
        // The light sample of an in-place vertex, taken where the cache did not end the path.
        if ((io.flags & PV_NEE) != 0u)
        {
            gathered += throughput * io.nee;
            if (litePath) liteL += liteSuffix * io.neeLite;
        }
        // A subsurface walk moved the vertex to its exit (io.pos), one depth further along the path.
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
            // What was gathered so far is the pixel's; from here on the path gathers relative to
            // the deferred scatter, and the head of the record is final.
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

        // Russian roulette scales surviving paths after the configured depth (deferred scatters
        // are replayed by the material pass).
        if (!(captured && kind == DV_KIND_BSDF) && PtPsDepth(ps) >= (uint)pt_rrStartDepth)
        {
            uint sRr = RcBounceSeed(PtPathSeed(pixel, s), PtPsDepth(ps), RC_STREAM_RR);
            const float survivalProb = max(min(1.0f, Luma(throughput)), 0.05f);
            if (RandomFloatSingle(sRr) >= survivalProb) break;
            throughput /= survivalProb;
            if (litePath) liteSuffix *= rcp(survivalProb);
        }

        // ---- next ray ----
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
