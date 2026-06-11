//====================================
//TRACE PAYLOAD AND HIT INFO
//====================================
//raygen uses SER HitObject path, stubs still need a nominal payload
struct [raypayload] TracePayload
{
    uint dummy : read(caller) : write(caller);
};

//no-medium sentinel for raygen absorption
static const uint MEDIUM_INVALID = 0xFFFFFFFFu;

struct HitInfo {
    float3 hitPos;
    float3 hitNormal;
    bool   backface;
    uint   lightID;
    float2 uv;
};


#ifdef ENABLE_RAY_QUERY_INLINE


//====================================
//RAY ORIGIN OFFSET, RTG CH 6
//====================================
//ULP-aware offset for self-intersection avoidance
static const float RTG_ORIGIN      = 1.0f / 32.0f;
static const float RTG_FLOAT_SCALE = 1.0f / 65536.0f;
static const float RTG_INT_SCALE   = 256.0f;

inline float3 offset_ray(float3 p, float3 n)
{
    int3 of_i = int3(RTG_INT_SCALE * n.x, RTG_INT_SCALE * n.y, RTG_INT_SCALE * n.z);
    float3 p_i = float3(
        asfloat(asint(p.x) + ((p.x < 0) ? -of_i.x : of_i.x)),
        asfloat(asint(p.y) + ((p.y < 0) ? -of_i.y : of_i.y)),
        asfloat(asint(p.z) + ((p.z < 0) ? -of_i.z : of_i.z)));
    return float3(
        abs(p.x) < RTG_ORIGIN ? p.x + RTG_FLOAT_SCALE * n.x : p_i.x,
        abs(p.y) < RTG_ORIGIN ? p.y + RTG_FLOAT_SCALE * n.y : p_i.y,
        abs(p.z) < RTG_ORIGIN ? p.z + RTG_FLOAT_SCALE * n.z : p_i.z);
}

//====================================
//RAY VALIDITY AND VISIBILITY
//====================================
//degenerate rays can hang BVH traversal, gate every TraceRay on this.
//The previous form only caught NaN/Inf and a 1e-12 magnitude floor, which
//let through finite but pathological rays that the BVH then crawled for
//seconds and tripped the Windows TDR. The current bounds are calibrated
//to the per call TMin (1e-5 / 1e-4) and the scene's bounded extent.
inline bool IsRayValid(float3 origin, float3 direction, float tMax)
{
    if (any(isnan(direction)) || any(isinf(direction))) return false;
    if (any(isnan(origin))    || any(isinf(origin)))    return false;

    //direction must be (approximately) unit. The old bound dot >= 1e-12
    //accepted |dir|=1e-6 which makes 1/dir blow up inside BVH slab tests
    //and pushes traversal into pathological numerical territory. Keep a
    //wide band so a slightly drifted normalize still passes, reject only
    //rays that are not order unity.
    const float d2 = dot(direction, direction);
    if (d2 < 0.25f || d2 > 4.0f) return false;

    //tMax must clear the actual TMin used downstream (1e-5 in the main
    //path tracer, 1e-4 in IsVisible). Coincident NEE lights have been
    //seen to feed in tMax = 0 here.
    if (tMax <= 1e-4f) return false;

    //origins well past the planet diameter (~1.3e7 m) lose FP32 precision
    //in BVH traversal. With the floating origin the camera is near zero,
    //but bounce rays from far-side terrain can reach ~2-3x planet radius.
    if (any(abs(origin) > 5.0e7f)) return false;

    return true;
}

//returns false for degenerate inputs, zero radiance safe default.
//P is an exact surface position (the reuse passes' reconnection shadow
//rays), so nudge the origin off the surface along the side the ray leaves
//- mirroring raygen's NEE shadow-origin pattern. The bare 1e-4 TMin alone
//does not clear fp32 ulp error at world-scale coordinates, and grazing
//connections self-intersected, silently killing valid reuse.
inline bool IsVisible(float3 P, float3 N_geo, float3 direction, float tMax)
{
    float3 origin = offset_ray(P, dot(direction, N_geo) >= 0.0f ? N_geo : -N_geo);

    if (!IsRayValid(origin, direction, tMax)) return false;

    RayDesc ray;
    ray.Origin    = origin;
    ray.Direction = direction;
    ray.TMin      = 0.0001f;
    ray.TMax      = tMax;

    RayQuery<RAY_FLAG_SKIP_CLOSEST_HIT_SHADER
       | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
       | RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();
    return q.CommittedStatus() == COMMITTED_NOTHING;
}

//pre-offset origin variant, skips per-surface normal lookup
inline bool IsVisibleOffset(float3 origin, float3 direction, float tMax)
{
    if (!IsRayValid(origin, direction, tMax)) return false;

    RayDesc ray;
    ray.Origin    = origin;
    ray.Direction = direction;
    ray.TMin      = 0.0001f;
    ray.TMax      = tMax;

    RayQuery<RAY_FLAG_SKIP_CLOSEST_HIT_SHADER
       | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
       | RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();
    return q.CommittedStatus() == COMMITTED_NOTHING;
}

//env-miss visibility: a shadow ray from a surface point toward a far (sky)
//direction. The origin is nudged off the surface along N (offset_ray) so it
//does not self-intersect; tMax is the sky distance (RAY_TMAX_PLANET). The
//ReSTIR GI passes (Pass_temp_gi / Pass_spat_gi_shift) call this for
//MATID_ENV_MISS reservoir samples. 'instID' fed the planet-era 3-arg
//offset_ray and is now unused - kept so those call sites compile unchanged.
inline bool IsVisibleEnvMiss(float3 P, float3 N, float3 direction, float tMax, uint instID)
{
    return IsVisibleOffset(offset_ray(P, N), direction, tMax);
}

//====================================
//NORMAL CLAMP TO VIEW AND REFLECTION
//====================================
inline float3 ClampNormalToViewAndReflection(float3 N, float3 V, float3 Ng, float epsView, float epsRefl)
{
    float3 Vn  = normalize(V);
    float3 Nn  = normalize(N);
    float3 NGn = normalize(Ng);

    //make sure the surface is hittable by V
    float a = dot(Nn, Vn);
    if (a < epsView)
    {
        float3 Nperp = Nn - a * Vn;
        float  len2  = dot(Nperp, Nperp);

        //degenerate, N ~ +/-V
        float3 u_any;
        {
            float3 t = (abs(Vn.x) > 0.5f) ? float3(-Vn.y, Vn.x, 0.0f) : float3(0.0f, -Vn.z, Vn.y);
            u_any = normalize(cross(Vn, t));
        }
        float3 u = (len2 < 1e-12f) ? u_any : (Nperp * rsqrt(len2));

        float s_view = sqrt(saturate(1.0f - epsView * epsView));
        Nn = normalize(s_view * u + epsView * Vn);
    }

    if (dot(Nn, NGn) < 0.0f)
        Nn = normalize(Nn - 2.0f * dot(Nn, NGn) * NGn);

    //quick reflection test
    {
        float3 R = reflect(-Vn, Nn);
        if (dot(R, NGn) >= epsRefl)
            return Nn;
    }

    //ensure reflected ray has coverage above normal
    float  c     = dot(Vn, NGn);
    float3 Ngp   = NGn - c * Vn;
    float  ngp2  = dot(Ngp, Ngp);
    float  m     = (ngp2 > 0.0f) ? rsqrt(ngp2) * ngp2 : 0.0f;
    float3 u;
    if (ngp2 > 1e-16f) {
        u = Ngp * rsqrt(ngp2);
    } else {
        float3 t = (abs(Vn.x) > 0.5f) ? float3(-Vn.y, Vn.x, 0.0f) : float3(0.0f, -Vn.z, Vn.y);
        u = normalize(cross(Vn, t));
    }

    a = saturate(dot(Nn, Vn));
    float thetaStart = acos(a);
    float thetaMax   = acos(saturate(epsView));

    float alpha = atan2(sqrt(saturate(1.0f - c*c)), c);
    float delta = acos(clamp(epsRefl, -1.0f, 1.0f));

    float L = 0.5f * (alpha - delta);
    float U = 0.5f * (alpha + delta);

    //intersect with [0, thetaMax]
    float Lc = max(0.0f, L);
    float Uc = min(thetaMax, U);

    float thetaTarget;
    if (Lc <= Uc)
    {
        thetaTarget = clamp(thetaStart, Lc, Uc);
    }
    else
    {
        thetaTarget = 0.0f;
    }

    float aT = cos(thetaTarget);
    float sT = sqrt(saturate(1.0f - aT * aT));
    float3 Nopt = aT * Vn + sT * u;

    if (dot(Nopt, NGn) < 0.0f)
        Nopt = normalize(Nopt - 2.0f * dot(Nopt, NGn) * NGn);
    else
        Nopt = normalize(Nopt);

    return Nopt;
}

//====================================
//TEXTURE EVALUATION
//====================================
float3 EvaluateAlbedo(uint matID, float2 uv, uint level)
{
    float3 albedo = LoadKd_rgb(matID);
    const int texID = LoadAlbedoTexID(matID);
    if (texID != -1)
    {
        float2 albedoUV = uv * LoadAlbedoUVScale(matID);
        Texture2D<float4> tex = ResourceDescriptorHeap[texID];
        albedo = tex.SampleLevel(g_sampler, albedoUV, level).rgb;
    }
    return albedo;
}

float2 EvaluatePBRProperties(uint matID, float2 uv, uint level)
{
    const float4 pbr4 = LoadPrPmPsPc(matID);
    float2 pbrProps = pbr4.xy;

    const int rmaID = LoadRmaTexID(matID);
    if (rmaID != -1)
    {
        float2 rmaUV = uv * LoadRmaUVScale(matID);
        Texture2D<float4> tex = ResourceDescriptorHeap[rmaID];
        float4 rmaSample = tex.SampleLevel(g_sampler, rmaUV, level);

        pbrProps.x = rmaSample.g;
        pbrProps.y = rmaSample.b;
    }
    return pbrProps;
}

//raygen passes depth to drop detail on deeper bounces
inline void RefetchMaterial(uint matID, float2 uv, out float3 localKd, out float localPr, out float localPm, uint level = 0)
{
    localKd = EvaluateAlbedo(matID, uv, level);
    float2 pbr = EvaluatePBRProperties(matID, uv, level);
    localPr = pbr.x;
    localPm = pbr.y;
}

//====================================
//CUSTOM TRACE RAY
//====================================
inline dx::HitObject TraceRay_Custom(
    RaytracingAccelerationStructure SceneBVH,
    RayDesc ray,
    uint rayFlags = RAY_FLAG_NONE,
    uint instanceMask = 0xFF)
{

    TracePayload payload = (TracePayload)0;
    //RayContribution=0, MultiplierForGeometry=1 (opaque vs alpha hit group per geometry), MissIndex=0
    dx::HitObject hitObj = dx::HitObject::TraceRay(SceneBVH, rayFlags, instanceMask, 0, 1, 0, ray, payload);

    uint hint = hitObj.IsHit()?1:0;
    dx::MaybeReorderThread(hitObj, hint, 1);
    return hitObj;
}


//====================================
//MISS EVALUATION
//====================================
inline float3 EvalMissState(float3 rayDir, float3 sunDisk)
{
    return EvaluateSky(rayDir);
}

//====================================
//SURFACE STATE EVALUATION
//====================================
HitInfo EvalSurfaceState(
    uint   instID,
    uint   primID,
    float2 bc2,
    float3 origin,
    uint   level
)
{
    //data gather
    const uint baseI      = instanceProps[instID].indexBase;
    const uint baseM      = instanceProps[instID].materialBase;
    const uint materialID = materialIDs[baseM + primID];

    const uint i0 = indices[baseI + 3u * primID + 0u];
    const uint i1 = indices[baseI + 3u * primID + 1u];
    const uint i2 = indices[baseI + 3u * primID + 2u];

    const float3 p0 = BTriVertex[i0].vertex;
    const float3 p1 = BTriVertex[i1].vertex;
    const float3 p2 = BTriVertex[i2].vertex;

    const float2 uv0 = (float2)BTriVertex[i0].texCoord;
    const float2 uv1 = (float2)BTriVertex[i1].texCoord;
    const float2 uv2 = (float2)BTriVertex[i2].texCoord;

    const uint pn0 = BTriVertex[i0].packedNormal;
    const uint pn1 = BTriVertex[i1].packedNormal;
    const uint pn2 = BTriVertex[i2].packedNormal;

    //local geometry
    const float b1 = bc2.x;
    const float b2 = bc2.y;
    const float b0 = 1.0f - b1 - b2;

    const float3 p_local = p0 * b0 + p1 * b1 + p2 * b2;
    const float2 uv      = uv0 * b0 + uv1 * b1 + uv2 * b2;

    //face normal, degeneracy test
    const float3 e1_local = p1 - p0;
    const float3 e2_local = p2 - p0;
    const float3 faceN_un = cross(e1_local, e2_local);
    const float  faceLen2 = dot(faceN_un, faceN_un);

    const float3 flatN_obj =
        (faceLen2 > 1e-20f) ? (faceN_un * rsqrt(faceLen2)) : float3(0.0f, 0.0f, 1.0f);

    //shading normal. PLANET (terrain + rocks: _pad[1] != 0) uses the FLAT
    //per-triangle geometric normal - the baked vertex normals are ignored, so
    //the surface reads as faceted triangles (the heightmap-gradient vertex
    //normals looked poor on the low-relief terrain). Everything else uses the
    //usual octahedral vertex normals with degenerate/flip guards, interpolated.
    //(For terrain, the runtime detail-normal below still rides on top of this
    //flat base; set PTD_STRENGTH = 0 in procedural_terrain_v8.hlsli for pure flat.)
    float3 n_local;
    [branch]
    if (instanceProps[instID]._pad[1] != 0u)
    {
        n_local = flatN_obj;
    }
    else
    {
        float3 n0 = UnpackNormal_INT(pn0);
        float3 n1 = UnpackNormal_INT(pn1);
        float3 n2 = UnpackNormal_INT(pn2);

        //replace degenerate/flipped normals with flat
        {
            const float n0l2 = dot(n0, n0);
            if (n0l2 < EPSILON) n0 = flatN_obj;
            else {
                const float inv0 = rsqrt(n0l2);
                if (dot(n0 * inv0, flatN_obj) <= 0.4f) n0 = flatN_obj;
            }

            const float n1l2 = dot(n1, n1);
            if (n1l2 < EPSILON) n1 = flatN_obj;
            else {
                const float inv1 = rsqrt(n1l2);
                if (dot(n1 * inv1, flatN_obj) <= 0.4f) n1 = flatN_obj;
            }

            const float n2l2 = dot(n2, n2);
            if (n2l2 < EPSILON) n2 = flatN_obj;
            else {
                const float inv2 = rsqrt(n2l2);
                if (dot(n2 * inv2, flatN_obj) <= 0.4f) n2 = flatN_obj;
            }
        }

        n_local = n0 * b0 + n1 * b1 + n2 * b2;
        n_local *= rsqrt(max(dot(n_local, n_local), 1e-20f));

        //force same hemisphere as geometric normal
        if (dot(n_local, flatN_obj) < 0.0f)
        {
            n_local = n_local - 2.0f * dot(n_local, flatN_obj) * flatN_obj;
            n_local *= rsqrt(max(dot(n_local, n_local), 1e-20f));
        }
    }

    //transform
    float3 posW;
    float3 normW;
    float3 geoNormW;
    float3 tangentW_geom;

    {
        const float4x4 M = instanceProps[instID].objectToWorld;
        const float3x3 R = (float3x3)M;

        posW     = mul(M, float4(p_local, 1.0f)).xyz;
        normW    = mul(R, n_local);
        normW   *= rsqrt(max(dot(normW, normW), 1e-20f));

        geoNormW = mul(R, flatN_obj);
        geoNormW *= rsqrt(max(dot(geoNormW, geoNormW), 1e-20f));

        //tangent from edges + UV deltas
        const float2 dUV1 = uv1 - uv0;
        const float2 dUV2 = uv2 - uv0;

        const float det = dUV1.x * dUV2.y - dUV1.y * dUV2.x;
        const float invDet = (abs(det) > 1e-8f) ? rcp(det) : 0.0f;

        const float3 tanO = (e1_local * dUV2.y - e2_local * dUV1.y) * invDet;

        tangentW_geom = mul(R, tanO);
        tangentW_geom *= rsqrt(max(dot(tangentW_geom, tangentW_geom), 1e-20f));
    }

    //material and texturing
    HitInfo hit = (HitInfo)0.0f;
    hit.uv = uv;

    //normal mapping
    const int normalTexID = LoadNormalTexID(materialID);
    [branch]
    if (normalTexID != -1)
    {
        const float2 normalUV = uv * LoadNormalUVScale(materialID);

        //Gram-Schmidt tangent against shading normal
        float3 tangentW = tangentW_geom - dot(tangentW_geom, normW) * normW;
        tangentW *= rsqrt(max(dot(tangentW, tangentW), 1e-20f));

        const float3 bitangentW = cross(normW, tangentW);

        Texture2D<float4> nTex = ResourceDescriptorHeap[normalTexID];
        const float3 n_tan =
            nTex.SampleLevel(g_sampler, normalUV, level).xyz * 2.0f - 1.0f;

        //apply TBN without materializing float3x3
        normW = n_tan.x * tangentW + n_tan.y * bitangentW + n_tan.z * normW;
        normW *= rsqrt(max(dot(normW, normW), 1e-20f));
    }

    //PLANET: runtime micro-detail normal (pebbles + sand) for terrain hits.
    //Normal-only bump finer than the baked mesh; world-stable + cm-precise via
    //the integer/fractional origin split. No-op for scene meshes (_pad[0]==0).
    //Done before the view clamp below so the clamp keeps the perturbed normal
    //valid for the ray to proceed.
    [branch]
    if (instanceProps[instID]._pad[0] != 0u)
    {
        const float hitDist = length(posW - origin);
        if (hitDist < PTD_MAX_DIST)
        {
            //absolute world = posW (floating-origin-shifted) + sceneOriginWorld,
            //carried as (integer, fractional) so the planet-scale magnitude
            //never forms a lossy fp32 sum.
            const int3   detOrigin = int3(round(sceneOriginWorld));
            const float3 detLocal  = posW + (sceneOriginWorld - float3(detOrigin));
            const float  detFoot   = max(hitDist * PTD_PIXEL_ANGLE, 1e-4f);
            normW = TerrainDetailNormal(detOrigin, detLocal, detFoot, normW, PTD_STRENGTH);
        }
    }

    //finalize
    float3 viewDir = posW - origin;
    viewDir *= rsqrt(max(dot(viewDir, viewDir), 1e-20f));

    const bool   isBackface      = (dot(viewDir, geoNormW) > 0.0f);
    const float3 geoNormOriented = isBackface ? -geoNormW : geoNormW;

    hit.hitPos    = posW;
    hit.hitNormal = isBackface ? -normW : normW;
    hit.backface  = isBackface;

    //clamp normal so ray can proceed
    {
        const float3 Vw = -viewDir;
        hit.hitNormal = ClampNormalToViewAndReflection(hit.hitNormal, Vw, geoNormOriented, 0.005f, 0.02f);
    }

    const uint baseL        = instanceProps[instID].triToLightBase;
    const uint frontLightID = gTriToLightId[baseL + primID];
    hit.lightID = isBackface ? 0xFFFFFFFFu : frontLightID;

    return hit;
}


//====================================
//FAST EMISSION AND MATERIAL LOOKUPS
//====================================
inline float3 GetEmissionFast(in uint instID, in uint primID)
{
    uint base = instanceProps[instID].triToLightBase;
    uint lightID = gTriToLightId[base + primID];
    if (lightID == 0xFFFFFFFF)
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    return g_EmissiveTriangles[lightID].emission * GLOBAL_EMISSION_STRENGTH;
}

inline uint GetMatIDFast(in uint instID, in uint primID){
    const uint baseI = instanceProps[instID].indexBase;
    const uint baseM = instanceProps[instID].materialBase;
    return materialIDs[baseM + primID];
}

#include "SurfaceVertex_v8.hlsli"

#endif
