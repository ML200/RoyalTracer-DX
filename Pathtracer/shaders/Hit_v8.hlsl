#include "Includes_v8.hlsli"
#include "PtVertex_v8.hlsli"

// Vertex of the split path tracer, invoked by the trace raygen after reordering. The primary vertex
// arrives through a short retrace of the camera hit and takes its surface from the camera record;
// every other vertex evaluates the surface and material here. Both feed one vertex routine.
[shader("closesthit")]
void ClosestHit(inout TracePayload payload, in BuiltInTriangleIntersectionAttributes attr)
{
    PtVertexIO io = PtIoFromPayload(payload);
    const uint2  pixel    = DispatchRaysIndex().xy;
    const uint   pixelIdx = MapPixelID(uint2(IMG_W, IMG_H), pixel);
    const uint   depth    = (io.flags >> PV_IN_DEPTH_SHIFT) & 0x7Fu;
    const float3 rayDir   = WorldRayDirection();
    const uint   instID   = InstanceID();
    const uint   primID   = FlatPrimID(instID, GeometryIndex(), PrimitiveIndex());

    HitContext ctx = (HitContext)0;
    float3 geoN      = 0.0f;
    bool   flipIOR   = false;
    uint   presetOut = 0u;
    bool   shade     = true;

    if (depth == 1u)
    {
        // The camera pass resolved the primary surface (including primary surface replacement);
        // only the geometric normal comes from the retraced triangle.
        const SDRecord sd = load_SD(g_sample_current, pixelIdx);
        float2 pIors; uint pMedium; float3 pAbsorb;
        load_rg_primaryExtra(pixelIdx, pIors, pMedium, pAbsorb);
        geoN = CandidateGeoNormalW(instID, primID);
        if (dot(geoN, sd.n1_s) < 0.0f) geoN = -geoN;

        ctx.hitPos         = sd.x1;
        ctx.hitNormal      = sd.n1_s;
        ctx.matID          = sd.matID;
        ctx.instID         = sd.instID;
        ctx.backface       = (sd.flags & SD_FLAG_BACKFACE) != 0u;
        ctx.hitLocalKd     = (half3)sd.Kd;
        ctx.hitLocalPr     = (half)sd.Pr;
        ctx.hitLocalPm     = (half)sd.Pm;
        ctx.iors           = (half2)pIors;
        ctx.mediumMatID    = pMedium;
        ctx.absorptionTint = (half3)pAbsorb;
        flipIOR = pMedium != MEDIUM_INVALID;
    }
    else
    {
        const uint    matID = GetMatIDFast(instID, primID);
        const HitInfo hinfo = EvalSurfaceStateDir(instID, primID, attr.barycentrics, rayDir, depth - 1u);
        const float3  emission = (hinfo.lightID != 0xFFFFFFFFu)
            ? g_EmissiveTriangles[hinfo.lightID].emission * GLOBAL_EMISSION_STRENGTH
            : float3(0, 0, 0);
        if (any(emission > 0.0f))
        {
            const bool pending   = (io.flags & PV_IN_PENDING) != 0u;
            const bool immediate = (io.flags & PV_IN_IMMEDIATE) != 0u;
            if (pending && immediate)
            {
                // MIS against the deferred light sample is resolved by the light and material passes.
                DvEmitter e;
                e.lightID = hinfo.lightID;
                e.inst = instID;
                e.pos = hinfo.hitPos;
                e.n = hinfo.hitNormal;
                DvStoreEmitter(pixelIdx, e);
                io.flags = PV_EMITTER_DEFERRED;
            }
            else
            {
                io.flags = PV_EMITTER;
                io.auxPk = PvPackRadiance(emission);
            }
            shade = false;
        }
        else if ((io.flags & PV_IN_LAST) != 0u)
        {
            io.flags = PV_TERMINATE;
            shade = false;
        }
        else
        {
            const float hitT         = RayTCurrent();
            const float matNi        = LoadNi(matID);
            const bool  transmissive = LoadKd_w(matID) < 1.0f - EPSILON;
            flipIOR = hinfo.backface && transmissive && !LoadIsThinGlass(matID);
            float3 hitLocalKd; float hitLocalPr, hitLocalPm;
            RefetchMaterial(matID, hinfo.uv, hitLocalKd, hitLocalPr, hitLocalPm, depth - 1u);

            if (depth == 2u && (io.flags & PV_IN_LITE_VERTEX) != 0u)
            {
                // The first bounce parks the reuse point; its broad-lobe share is applied by the material pass.
                const float3 liteNy = dot(hinfo.geometricNormal, -rayDir) < 0.0f
                    ? -hinfo.geometricNormal : hinfo.geometricNormal;
                LiteParkPointStore(pixelIdx, WorldToObjectPos(instID, hinfo.hitPos), instID,
                    PackNormal(WorldToObjectNrm(instID, liteNy)), DvLoadScatterPdf(pixelIdx));
                presetOut |= PV_LITE_PARKED;
            }

            if ((io.flags & PV_IN_SPREAD) != 0u)
                io.spread += hitT * sqrt(min(16.0f, rcp(max(io.pdf * abs(dot(hinfo.geometricNormal, -rayDir)), 1e-6f))));
            const float regularize = io.spread > 0.0f ? PT_REGULARIZE_ROUGHNESS : 0.0f;

            geoN = hinfo.geometricNormal;
            ctx.hitPos         = hinfo.hitPos;
            ctx.hitNormal      = hinfo.hitNormal;
            ctx.matID          = matID;
            ctx.instID         = instID;
            ctx.backface       = hinfo.backface;
            ctx.hitLocalKd     = (half3)hitLocalKd;
            ctx.hitLocalPr     = (half)max(hitLocalPr, regularize);
            ctx.hitLocalPm     = (half)hitLocalPm;
            ctx.iors           = (half2)(flipIOR ? float2(matNi, 1.0f) : float2(1.0f, matNi));
            ctx.mediumMatID    = flipIOR ? matID : MEDIUM_INVALID;
            ctx.absorptionTint = (half3)(flipIOR ? CalculateAbsorptionThroughput(LoadTf(matID), hitT) : float3(1, 1, 1));
        }
    }

    if (shade) PtVertexShade(io, ctx, geoN, rayDir, flipIOR, pixel, pixelIdx, presetOut);
    PtIoToPayload(io, payload);
}
