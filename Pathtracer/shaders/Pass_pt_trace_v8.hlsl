#include "Includes_v8.hlsli"
#include "PtVertex_v8.hlsli"
#include "OceanVolume.hlsli"

// Half-length of the ray that hands the camera hit to the closest-hit shader, relative to the
// distance from the scene origin (positions are stored relative to it).
static const float PT_RETRACE_SCALE = 1e-4f;
static const float PT_RETRACE_MIN   = 1e-3f;

// Drive one sample of the pixel path. Every vertex is shaded here after reordering on the hit
// object; the path loop keeps the path state, applies the compact results and finalizes the
// deferred record for the light and material passes.
[shader("raygeneration")]
void Pass_pt_trace_v8()
{
    const uint2 pixel   = DispatchRaysIndex().xy;
    const uint2 imgSize = uint2(IMG_W, IMG_H);
    if (any(pixel >= imgSize)) return;
    const uint pixelIdx = MapPixelID(imgSize, pixel);
    const uint s = PT_SAMPLE_INDEX;

    if (load_flagsWord(g_sample_current, pixelIdx) & SD_FLAG_NOBOUNCE)
    {
        DvStoreInfo(pixelIdx, 0u);
        return;
    }
    if (s == 0u) gScratchPing[uint3(pixel, 2)] = 0.0f;

    const uint maxBounces = pt_maxBounces;
    const uint pathSeed   = PtPathSeed(pixel, s);

    float3 throughput = float3(1, 1, 1);   // absolute until the deferred vertex, relative afterwards
    float  prev_pdf   = 1.0f;
    float  pathSpread = 0.0f;
    float3 liteL      = 0.0f;
    float3 liteSuffix = 0.0f;
    uint   ps = PtPsInit(s);
    float3 total = 0.0f;

    // Deferred vertex state.
    bool   pending   = false;
    bool   immediate = false;
    float3 pendingT  = 0.0f;
    float  pendingSpread = 0.0f;
    uint   pendingPs = 0u;
    float3 relL      = 0.0f;
    uint   info      = 0u;
    DvMiss missEnd = (DvMiss)0;

    // The camera pass resolved the primary vertex; a short ray through it gives the reorder a hit
    // object, so the primary vertices are sorted like every other hit. The primary shading takes
    // the camera record whether or not this ray hits.
    float3 pos    = load_x1(g_sample_current, pixelIdx);
    const float3 toPrimary = pos - InitOrigin();
    float  pathDist = length(toPrimary);   // path length: with the pixel cone it sizes the texture footprint
    float3 rayDir   = toPrimary / max(pathDist, 1e-6f);
    const float retrace = max(PT_RETRACE_MIN, PT_RETRACE_SCALE * length(pos));
    RayDesc ray;
    ray.Origin    = pos - rayDir * retrace;
    ray.Direction = rayDir;
    ray.TMin      = 0.0f;
    ray.TMax      = 2.0f * retrace;
    // A degenerate camera record (a non-finite or zero-length primary segment) makes this ray
    // invalid; the checked trace answers it with a miss, and the primary shading rejects the
    // record itself.
    uint inFlags = PvInputFlags(ps, false, false, false);
    uint cameraSeed = initRandomData(pixel, uint2(8,4), time, 1u);
    float3 cameraOrigin, cameraDirection;
    InitCameraRayDoF(pixel,imgSize,cameraSeed,cameraOrigin,cameraDirection);
    const uint primaryMat = load_matID(g_sample_current,pixelIdx);
    const bool cameraWater = LoadIsOceanMaterial(primaryMat) ?
        (load_flagsWord(g_sample_current,pixelIdx) & SD_FLAG_BACKFACE) != 0u : OceanPointInside(cameraOrigin);
    if (cameraWater) inFlags |= PV_IN_WATER_MEDIUM;
    float3 prevN = float3(0.0f, 0.0f, 1.0f);   // shading normal of the vertex the ray left

    [loop]
    for (;;)
    {
        // ---- trace, reorder, shade ----
        const uint depth = PtPsDepth(ps);
        dx::HitObject hitObj = TraceRayChecked(SceneBVH, RAY_FLAG_FORCE_OMM_2_STATE, 0xFF, ray);
        dx::MaybeReorderThread(hitObj);
        // The primary segment is composed once in the final pass, also covering
        // camera misses and directly visible emitters that bypass this path loop.
        if (depth > 1u && (inFlags & PV_IN_WATER_MEDIUM) != 0u) {
            uint sVolume = RcBounceSeed(pathSeed, depth, 0x57415452u);
            float3 waterT, waterL;
            OceanIntegrateVolume(ray.Origin, rayDir, hitObj.IsHit() ? hitObj.GetRayTCurrent() : ray.TMax,
                sVolume, waterT, waterL);
            if (pending) relL += throughput*waterL; else total += throughput*waterL;
            if (depth >= 2u && (ps & PT_PS_LITE_VERTEX) != 0u) {
                liteL += liteSuffix*waterL;
                liteSuffix *= waterT;
            }
            throughput *= waterT;
            // A deferred emitter replay has no slot for segment attenuation.
            // Resolve this endpoint in place using the now-attenuated throughput.
            immediate = false;
            inFlags &= ~PV_IN_IMMEDIATE;
        }
        PtVertexIO io;
        io.flags = inFlags; io.pdf = prev_pdf; io.spread = pathSpread; io.dist = pathDist;
        io.dirPk = 0u; io.nPk = 0u; io.color = 0.0f; io.auxPk = 0u; io.nee = 0.0f; io.neeLite = 0.0f;
        if (depth == 1u)
            PtShadePrimary(io, rayDir, pixel, pixelIdx);
        else if (hitObj.IsHit())
        {
            BuiltInTriangleIntersectionAttributes attr;
            hitObj.GetAttributes(attr);
            PtShadeHit(io, hitObj.GetInstanceID(), hitObj.GetGeometryIndex(), hitObj.GetPrimitiveIndex(),
                attr.barycentrics, hitObj.GetRayTCurrent(), rayDir, pos, prevN, pixel, pixelIdx);
        }
        else
            PtShadeMiss(io, ray.Origin, rayDir);
        if (depth == 1u)
        {
            if (LITE_ENABLED && s == 0u && (io.flags & PV_PRIMARY_LITE) == 0u) LiteMarkEmpty(pixelIdx);
            ps = PtPsWith(ps, PT_PS_LITE_VERTEX, (io.flags & PV_PRIMARY_LITE) != 0u);
        }
        else if (hitObj.IsHit())
            pos = ray.Origin + rayDir * hitObj.GetRayTCurrent();

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
            if (pending)
            {
                relL += throughput * envL;
                if (litePath) liteL += liteSuffix * envL;
                missEnd.dir = rayDir;
                missEnd.skySun = sky + sunRad;
                missEnd.candMis = sunMis;
                const bool liteGen = PtPsDepth(pendingPs) == 1u && liteVertex;
                info |= (DV_END_MISS << DV_END_SHIFT) | ((immediate && liteGen) ? DV_LITE_MISS : 0u);
            }
            else total += throughput * envL;
            break;
        }
        if (result == PV_EMITTER)
        {
            const float3 emission = PvUnpackRadiance(io.auxPk);
            if (pending) relL += throughput * emission; else total += throughput * emission;
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
            const float3 gathered = PvUnpackRadiance(io.auxPk);
            if (pending) relL += throughput * gathered; else total += throughput * gathered;
            if (litePath)
            {
                liteL += liteSuffix * gathered;
                liteSuffix = 0.0f;
                if (!any(throughput > 0.0f)) break;
            }
        }
        // The light sample of an in-place vertex, taken where the cache did not end the path.
        if ((io.flags & PV_NEE) != 0u)
        {
            if (pending) relL += throughput * io.nee; else total += throughput * io.nee;
            if (litePath) liteL += liteSuffix * io.neeLite;
        }
        // A subsurface walk moved the vertex to its exit, one depth further along the path.
        if ((io.flags & PV_SSS_WALKED) != 0u)
        {
            ps |= PT_PS_SSS;
            ps += 1u << PT_PS_DEPTH_SHIFT;
            if (PtPsGuideDepth(ps) < 15u) ps += 1u << PT_PS_GUIDE_SHIFT;
            pos = DvLoadSssExit(pixelIdx);
        }
        const uint kind = (io.flags >> PV_KIND_SHIFT) & DV_KIND_MASK;
        const bool captured = (io.flags & PV_CAPTURED) != 0u;
        if (captured)
        {
            pending = true;
            pendingPs = ps;
            pendingSpread = io.spread;
            info = DV_VALID | kind | (((io.flags >> PV_STRATEGY_SHIFT) & 3u) << DV_STRATEGY_SHIFT) |
                ((io.flags & PV_MIS_NONE) != 0u ? DV_MIS_NONE : 0u);
            pendingT = throughput * io.color;
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
        pathDist   = io.dist;
        if (result == PV_TERMINATE) break;

        // Russian roulette scales surviving paths after the configured depth (deferred scatters
        // are replayed by the material pass).
        if (!(captured && kind == DV_KIND_BSDF) && PtPsDepth(ps) >= (uint)pt_rrStartDepth)
        {
            uint sRr = RcBounceSeed(pathSeed, PtPsDepth(ps), RC_STREAM_RR);
            const float survivalProb = max(min(1.0f, Luma(throughput)), 0.05f);
            if (RandomFloatSingle(sRr) >= survivalProb) break;
            throughput /= survivalProb;
            if (litePath) liteSuffix *= rcp(survivalProb);
        }

        // ---- next ray ----
        rayDir = UnpackNormal(io.dirPk);
        const float3 vertexN = UnpackNormal(io.nPk);
        prevN = vertexN;
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
    }

    PtAddRadiance(pixel, total);
    if (!pending)
    {
        DvStoreInfo(pixelIdx, 0u);
        return;
    }
    DvStoreInfo(pixelIdx, info);
    DvPath path;
    path.T = pendingT;
    path.pathSpread = pendingSpread;
    path.relL = relL;
    path.ps = pendingPs | (ps & PT_PS_LITE_X2);
    DvStorePath(pixelIdx, path);
    if ((ps & PT_PS_LITE_X2) != 0u) DvStoreLiteL(pixelIdx, liteL);
    if (((info >> DV_END_SHIFT) & DV_END_MASK) == DV_END_MISS) DvStoreMiss(pixelIdx, missEnd);
}
