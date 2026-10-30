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

    // Absolute until the deferred vertex, then relative to its scatter. Radiance is not carried: it goes
    // to the pixel, or after the deferred vertex to its relL and reuse record, where it arises (PtGather).
    float3 throughput = float3(1, 1, 1);
    float  prev_pdf   = 1.0f;
    float  pathSpread = 0.0f;
    uint   pathCone   = 0u;     // lobe cone for the cache test (PtConePack)
    float3 liteSuffix = 0.0f;
    uint   ps = PtPsInit(s);

    // Deferred vertex state.
    bool   pending   = false;
    bool   immediate = false;
    uint   info      = 0u;

    // The first iteration shades the camera record, untraced.
    float3 pos = load_x1(g_sample_current, pixelIdx);   // the vertex the next ray leaves
    const float3 toPrimary = pos - InitOrigin();
    float  pathDist = length(toPrimary);   // for the texture footprint
    RayDesc ray;   // the direction lives here only; nothing past the trace needs it
    ray.Origin    = pos;
    ray.Direction = toPrimary / max(pathDist, 1e-6f);
    ray.TMin      = 0.0f;
    ray.TMax      = 0.0f;
    uint inFlags = PvInputFlags(ps, false, false, false);
    bool cameraRecordPending = !cameraWater;
    if (cameraWater) {
        uint cameraSeed = initRandomData(pixel,uint2(8,4),time,1u);
        InitCameraRayDoF(pixel,imgSize,cameraSeed,ray.Origin,ray.Direction);
        ray.TMin = 0.00001f; ray.TMax = RAY_TMAX_PLANET;
        pos = ray.Origin; pathDist = 0.0f;
        inFlags |= PV_IN_WATER_MEDIUM;
        if (LITE_ENABLED && s == 0u) LiteMarkEmpty(pixelIdx);
    }
    uint prevNPk = PackNormal(float3(0.0f, 0.0f, 1.0f));   // shading normal of the vertex the ray left
    uint waterSkips = 0u;   // wrong-side water facets passed on this segment

    [loop]
    for (;;)
    {
        const uint depth = PtPsDepth(ps);
        // The camera record is shaded through the primary miss record; later vertices are traced.
        dx::HitObject hitObj;
        uint reorderHint = 0x100u;
        if (cameraRecordPending)
        {
            const uint primaryInst = load_instID(g_sample_current, pixelIdx);
            // Only a sea surface needs the full shader (water NEE).
            hitObj = dx::HitObject::MakeMiss(RAY_FLAG_NONE,
                IS_OCEAN_INSTANCE(primaryInst) ? PT_MISS_PRIMARY : PT_MISS_PRIMARY_LEAN, ray);
            reorderHint = primaryInst & 0xffu;
        }
        else
        {
            hitObj = TraceRayChecked(SceneBVH, RAY_FLAG_FORCE_OMM_2_STATE, 0xFF, ray, PT_MISS_SKY);
            const OceanPathHit hit = OceanPathHitOf(hitObj, ray);
            // Pass up to 4 wrong-side water facets (OceanTracePathHitInMedium), one trace site.
            if (waterSkips < 4u && OceanFacetPassed(hit, ray, (inFlags & PV_IN_WATER_MEDIUM) != 0u))
            {
                ray.TMin = hit.distance + 1e-3f;
                ++waterSkips;
                continue;
            }
            waterSkips = 0u;
            if ((inFlags & PV_IN_WATER_MEDIUM) != 0u) {
                // Water in-scatter and attenuation (PtWaterSegment); only the first segment samples the
                // sun and lights.
                const bool sampled = (inFlags & PV_IN_WATER_SCATTERED) == 0u && depth <= (cameraWater ? 1u : 2u);
                PtWaterSegmentIO w;
                w.origin  = ray.Origin;
                w.dirPk   = PackNormal(ray.Direction);
                w.hitDist = hit.distance;
                w.bits    = depth | (sampled ? PT_WATER_SAMPLED : 0u);
                CallShader(PT_CALLABLE_WATER_SEGMENT, w);
                const float3 inScatter = PvUnpackRadiance(w.dirPk);
                PtGather(pixel, pixelIdx, pending, throughput * inScatter);
                if (depth >= 2u && (ps & (PT_PS_LITE_VERTEX | PT_PS_LITE_X2)) == (PT_PS_LITE_VERTEX | PT_PS_LITE_X2))
                    DvAddLiteL(pixelIdx, liteSuffix * inScatter);
                throughput *= w.origin;
                if ((ps & PT_PS_LITE_VERTEX) != 0u) liteSuffix *= w.origin;
                // Deferred emitter replay can't attenuate; resolve in place.
                immediate = false;
                inFlags &= ~PV_IN_IMMEDIATE;
                inFlags |= PV_IN_WATER_SCATTERED;
                if (!any(throughput > 0.0f)) break;
            }
            if (hit.hit) reorderHint = hit.instance & 0xffu;
            // Water state and sea hits need the full shader (Hit_v8.hlsl); the rest runs HitLean_v8.hlsl.
            if (hit.hit && (inFlags & PV_IN_WATER_ANY) == 0u && !IS_OCEAN_INSTANCE(hit.instance))
                hitObj.SetShaderTableIndex(PT_HIT_LEAN_RECORD);
        }
        // Keep outside the trace branch: reordering inside it hung the device.
#if PT_SER_REORDER
        dx::MaybeReorderThread(hitObj, reorderHint | (pending ? 0x200u : 0u), 10u);
#endif
        // ClosestHit(Lean), Miss or PtPrimaryShade(Lean). Filled after the reorder, so only the path
        // state rides through it; the outputs are the callee's.
        PtVertexIO io;
        io.flags = inFlags; io.pdf = prev_pdf; io.spread = pathSpread; io.dist = pathDist; io.cone = pathCone;
        io.pos = pos; io.nPk = prevNPk;
        dx::HitObject::Invoke(hitObj, io);
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
        // The reuse suffix's radiance counts once its point is parked (PT_PS_LITE_X2).
        bool liteRec = litePath && (ps & PT_PS_LITE_X2) != 0u;

        if (result == PV_MISS)
        {
            const float3 sky = io.color;
            const float3 sunRad = PvUnpackRadiance(io.auxPk);
            const float  sunMis = io.pdf;
            const float3 envL = sky + sunRad * sunMis;
            PtGather(pixel, pixelIdx, pending, throughput * envL);
            if (pending)
            {
                if (liteRec) DvAddLiteL(pixelIdx, liteSuffix * envL);
                DvStoreMiss(pixelIdx, io.dirPk, sky + sunRad, sunMis);
                info |= (DV_END_MISS << DV_END_SHIFT) | ((immediate && (info & DV_LITE_GEN) != 0u) ? DV_LITE_MISS : 0u);
            }
            break;
        }
        if (result == PV_EMITTER)
        {
            const float3 emission = PvUnpackRadiance(io.auxPk);
            PtGather(pixel, pixelIdx, pending, throughput * emission);
            if (liteRec) DvAddLiteL(pixelIdx, liteSuffix * emission);
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
            DvStoreLiteL(pixelIdx, 0.0f);
            ps |= PT_PS_LITE_X2;
            liteRec = litePath;
        }
        // A deferred light sample (Pass_pt_nee_v8) gets the weights it arrives with.
        if ((io.flags & PV_NEE_QUEUED) != 0u)
            PtNeeQueueStoreWeights(io.neePk, throughput, liteRec ? liteSuffix : 0.0f);
        if ((io.flags & PV_CACHE_HIT) != 0u)
        {
            const float3 cached = PvUnpackRadiance(io.auxPk);
            PtGather(pixel, pixelIdx, pending, throughput * cached);
            if (litePath)
            {
                if (liteRec) DvAddLiteL(pixelIdx, liteSuffix * cached);
                liteSuffix = 0.0f;
                if (!any(throughput > 0.0f)) break;
            }
        }
        if ((io.flags & PV_NEE) != 0u)
        {
            // A water light sample, shaded inline.
            PtGather(pixel, pixelIdx, pending, throughput * PvUnpackRadiance(io.neePk));
            if (liteRec) DvAddLiteL(pixelIdx, liteSuffix * PvUnpackRadiance(io.neeLitePk));
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
            // Radiance so far is in the pixel; relL gathers relative to the deferred scatter from here.
            DvStorePathRelL(pixelIdx, 0.0f);
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

        const float3 rayDir  = UnpackNormal(io.dirPk);
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
        inFlags = PvInputFlags(ps, pending, immediate,
            nextDepth >= maxBounces || (io.flags & PV_LAST_NEXT) != 0u);
        inFlags |= (io.flags & PV_WATER_DIRECT) != 0u ? PV_IN_WATER_DIRECT : 0u;
        inFlags |= (io.flags & PV_WATER_MEDIUM) != 0u ? PV_IN_WATER_MEDIUM : 0u;
        inFlags |= (io.flags & PV_WATER_SCATTERED) != 0u ? PV_IN_WATER_SCATTERED : 0u;
        inFlags |= (io.flags & PV_SUN_OWNED) != 0u ? PV_IN_SUN_OWNED : 0u;
    }

    if ((ps & PT_PS_LITE_X2) != 0u) info |= DV_LITE_X2;
    DvStoreInfo(pixelIdx, pending ? info : 0u);
}
