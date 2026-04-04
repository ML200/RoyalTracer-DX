// Minimal payload type
struct [raypayload] TracePayload
{
    uint dummy : read(caller) : write(caller);
};


#ifdef ENABLE_RAY_QUERY_INLINE

// RTG Ch. 6 — ULP-aware origin offset for self-intersection avoidance
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

// Shadow/visibility test with RTG origin offset and alpha testing
inline bool IsVisible(float3 P, float3 N_geo, float3 direction, float tMax)
{
    float3 origin = offset_ray(P, dot(direction, N_geo) >= 0.0f ? N_geo : -N_geo);

    RayDesc ray;
    ray.Origin    = origin;
    ray.Direction = direction;
    ray.TMin      = 0.001f;
    ray.TMax      = tMax;

    RayQuery<RAY_FLAG_SKIP_CLOSEST_HIT_SHADER | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, ray);

    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            uint   instID = q.CandidateInstanceID();
            uint   primID = FlatPrimID(instID, q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex());
            uint   matID  = materialIDs[instanceProps[instID].materialBase + primID];
            Material mat  = materials[matID];

            if (mat.albedoTexID < 0)
            {
                q.CommitNonOpaqueTriangleHit();
                continue;
            }

            uint baseI = instanceProps[instID].indexBase;
            uint i0 = indices[baseI + 3u * primID + 0u];
            uint i1 = indices[baseI + 3u * primID + 1u];
            uint i2 = indices[baseI + 3u * primID + 2u];

            float2 uv0 = (float2)BTriVertex[i0].texCoord;
            float2 uv1 = (float2)BTriVertex[i1].texCoord;
            float2 uv2 = (float2)BTriVertex[i2].texCoord;

            float2 bc  = q.CandidateTriangleBarycentrics();
            float  b0  = 1.0f - bc.x - bc.y;
            float2 uv  = uv0 * b0 + uv1 * bc.x + uv2 * bc.y;

            Texture2D<float4> tex = ResourceDescriptorHeap[mat.albedoTexID];
            float alpha = tex.SampleLevel(g_sampler, uv * mat.albedoUVScale, 0).a;

            if (alpha >= mat.alphaThreshold)
                q.CommitNonOpaqueTriangleHit();
        }
    }

    return (q.CommittedStatus() == COMMITTED_NOTHING);
}

inline float3 ClampNormalToViewAndReflection(float3 N, float3 V, float3 Ng, float epsView, float epsRefl)
{
    float3 Vn  = normalize(V);
    float3 Nn  = normalize(N);
    float3 NGn = normalize(Ng);

    // Make sure the surface defined by the interpolated normal is hittable by the view vector
    float a = dot(Nn, Vn); // cos between N and V
    if (a < epsView)
    {
        float3 Nperp = Nn - a * Vn;
        float  len2  = dot(Nperp, Nperp);

        // Degenerate: N ~ +/- V
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

    // Quick reflection test
    {
        float3 R = reflect(-Vn, Nn);
        if (dot(R, NGn) >= epsRefl)
            return Nn;
    }

    // Make sure that a potentially reflected ray has substantial coverage above the normal
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

    // Intersect with [0, thetaMax]
    float Lc = max(0.0f, L);
    float Uc = min(thetaMax, U);

    float thetaTarget;
    if (Lc <= Uc)
    {
        // If current θ inside intersection, keep it
        thetaTarget = clamp(thetaStart, Lc, Uc);
    }
    else
    {
        // No feasible θ within view constraint. Best-effort fallback
        thetaTarget = 0.0f;
    }

    // Rebuild N
    float aT = cos(thetaTarget);
    float sT = sqrt(saturate(1.0f - aT * aT));
    float3 Nopt = aT * Vn + sT * u;

    // Hemisphere fix
    if (dot(Nopt, NGn) < 0.0f)
        Nopt = normalize(Nopt - 2.0f * dot(Nopt, NGn) * NGn);
    else
        Nopt = normalize(Nopt);

    return Nopt;
}

float3 EvaluateAlbedo(in Material mat, float2 uv, uint level)
{
    float3 albedo = mat.Kd.rgb;
    if (mat.albedoTexID != -1)
    {
        float2 albedoUV = uv * mat.albedoUVScale;
        Texture2D<float4> tex = ResourceDescriptorHeap[mat.albedoTexID];
        albedo = tex.SampleLevel(g_sampler, albedoUV, level).rgb;
    }
    return albedo;
}

float2 EvaluatePBRProperties(in Material mat, float2 uv, uint level)
{
    float2 pbrProps;
    pbrProps.x = mat.Pr_Pm_Ps_Pc.x;
    pbrProps.y = mat.Pr_Pm_Ps_Pc.y;

    if (mat.rmaTexID != -1)
    {
        float2 rmaUV = uv * mat.rmaUVScale;
        Texture2D<float4> tex = ResourceDescriptorHeap[mat.rmaTexID];
        float4 rmaSample = tex.SampleLevel(g_sampler, rmaUV, level);

        pbrProps.x = rmaSample.g;
        pbrProps.y = rmaSample.b;
    }
    return pbrProps;
}
// Refetch material properties from textures using matID + UV (level 0)
inline void RefetchMaterial(uint matID, float2 uv, out float3 localKd, out float localPr, out float localPm)
{
    Material mat = materials[matID];
    localKd = EvaluateAlbedo(mat, uv, 0);
    float2 pbr = EvaluatePBRProperties(mat, uv, 0);
    localPr = pbr.x;
    localPm = pbr.y;
}

// =====================================================================================================================

inline dx::HitObject TraceRay_Custom(
    RaytracingAccelerationStructure SceneBVH,
    RayDesc ray,
    uint rayFlags = RAY_FLAG_NONE,
    uint instanceMask = 0xFF)
{

    TracePayload payload = (TracePayload)0;
    dx::HitObject hitObj = dx::HitObject::TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ray, payload);

    uint hint = hitObj.IsHit()?1:0;
    dx::MaybeReorderThread(hitObj, hint, 1);
    return hitObj;
}


inline float3 EvalMissState(float3 rayDir, float3 sunDisk)
{
    return EvaluateSky(rayDir);
}

HitInfo EvalSurfaceState(
    uint   instID,
    uint   primID,
    float2 bc2,
    float3 origin,
    uint   level
)
{
    // 1) DATA GATHER (keep indices/offsets short-lived)
    const uint baseI      = instanceProps[instID].indexBase;
    const uint baseM      = instanceProps[instID].materialBase;
    const uint materialID = materialIDs[baseM + primID];

    const uint i0 = indices[baseI + 3u * primID + 0u];
    const uint i1 = indices[baseI + 3u * primID + 1u];
    const uint i2 = indices[baseI + 3u * primID + 2u];

    // Load only required vertex fields (avoid whole-struct register bloat)
    const float3 p0 = BTriVertex[i0].vertex;
    const float3 p1 = BTriVertex[i1].vertex;
    const float3 p2 = BTriVertex[i2].vertex;

    const float2 uv0 = (float2)BTriVertex[i0].texCoord;
    const float2 uv1 = (float2)BTriVertex[i1].texCoord;
    const float2 uv2 = (float2)BTriVertex[i2].texCoord;

    const uint pn0 = BTriVertex[i0].packedNormal;
    const uint pn1 = BTriVertex[i1].packedNormal;
    const uint pn2 = BTriVertex[i2].packedNormal;

    // 2) LOCAL GEOMETRY
    const float b1 = bc2.x;
    const float b2 = bc2.y;
    const float b0 = 1.0f - b1 - b2;

    const float3 p_local = p0 * b0 + p1 * b1 + p2 * b2;
    const float2 uv      = uv0 * b0 + uv1 * b1 + uv2 * b2;

    // Face normal + degeneracy test without length()
    const float3 e1_local = p1 - p0;
    const float3 e2_local = p2 - p0;
    const float3 faceN_un = cross(e1_local, e2_local);
    const float  faceLen2 = dot(faceN_un, faceN_un);

    // Normalize via rsqrt; use fallback for degenerate triangle
    const float3 flatN_obj =
        (faceLen2 > 1e-20f) ? (faceN_un * rsqrt(faceLen2)) : float3(0.0f, 0.0f, 1.0f);

    // Shading normals (keep temporaries tight)
    float3 n0 = UnpackNormal_INT(pn0);
    float3 n1 = UnpackNormal_INT(pn1);
    float3 n2 = UnpackNormal_INT(pn2);

    // Replace degenerate / flipped / junk normals with flat normal
    // (avoid normalize() unless needed)
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

    // Interpolate and normalize shading normal
    float3 n_local = n0 * b0 + n1 * b1 + n2 * b2;
    n_local *= rsqrt(max(dot(n_local, n_local), 1e-20f));

    // Ensure shading normal points to same hemisphere as geometric normal
    if (dot(n_local, flatN_obj) < 0.0f)
    {
        n_local = n_local - 2.0f * dot(n_local, flatN_obj) * flatN_obj;
        n_local *= rsqrt(max(dot(n_local, n_local), 1e-20f));
    }

    // 3) TRANSFORM (only keep what is needed across later phases)
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

        // Geometric tangent in object space (from edges + UV deltas), then transform.
        const float2 dUV1 = uv1 - uv0;
        const float2 dUV2 = uv2 - uv0;

        const float det = dUV1.x * dUV2.y - dUV1.y * dUV2.x;
        const float invDet = (abs(det) > 1e-8f) ? rcp(det) : 0.0f;

        // Unnormalized object tangent
        const float3 tanO = (e1_local * dUV2.y - e2_local * dUV1.y) * invDet;

        tangentW_geom = mul(R, tanO);
        // Normalize (safe even if det==0 -> tanO==0)
        tangentW_geom *= rsqrt(max(dot(tangentW_geom, tangentW_geom), 1e-20f));
    }

    // 4) MATERIAL & TEXTURING
    const Material mat = materials[materialID];

    HitInfo hit = (HitInfo)0.0f;
    hit.uv = uv;

    // 5) NORMAL MAPPING (avoid TBN matrix; scope aggressively)
    [branch]
    if (mat.normalTexID != -1)
    {
        const float2 normalUV = uv * mat.normalUVScale;

        // Gram–Schmidt tangent against shading normal
        float3 tangentW = tangentW_geom - dot(tangentW_geom, normW) * normW;
        tangentW *= rsqrt(max(dot(tangentW, tangentW), 1e-20f));

        const float3 bitangentW = cross(normW, tangentW);

        Texture2D<float4> nTex = ResourceDescriptorHeap[mat.normalTexID];
        const float3 n_tan =
            nTex.SampleLevel(g_sampler, normalUV, level).xyz * 2.0f - 1.0f;

        // Apply TBN without materializing float3x3
        normW = n_tan.x * tangentW + n_tan.y * bitangentW + n_tan.z * normW;
        normW *= rsqrt(max(dot(normW, normW), 1e-20f));
    }

    // 6) FINALIZE HIT INFO
    float3 viewDir = posW - origin;
    viewDir *= rsqrt(max(dot(viewDir, viewDir), 1e-20f));

    const bool isBackface = (dot(viewDir, geoNormW) > 0.0f);

    hit.hitPos     = posW;
    hit.hitNormal  = isBackface ? -normW    : normW;
    hit.hitGNormal = isBackface ? -geoNormW : geoNormW;

    // Clamp (keep Vw scoped)
    {
        const float3 Vw = -viewDir;
        hit.hitNormal = ClampNormalToViewAndReflection(hit.hitNormal, Vw, hit.hitGNormal, 0.1f, 0.02f);
    }

    const uint baseL        = instanceProps[instID].triToLightBase;
    const uint frontLightID = gTriToLightId[baseL + primID];
    hit.lightID = isBackface ? 0xFFFFFFFFu : frontLightID;

    return hit;
}


// Cheap helper to check if hit triangle is a light and return emission for immediate processing
inline float3 GetEmissionFast(in uint instID, in uint primID)
{
    uint base = instanceProps[instID].triToLightBase;
    uint lightID = gTriToLightId[base + primID];
    if (lightID == 0xFFFFFFFF)
    {
        return float3(0.0f, 0.0f, 0.0f);
    }
    return g_EmissiveTriangles[lightID].emission;
}

inline uint GetMatIDFast(in uint instID, in uint primID){
    const uint baseI = instanceProps[instID].indexBase;
    const uint baseM = instanceProps[instID].materialBase;
    return materialIDs[baseM + primID];
}

// =====================================================================================================================
// SurfaceVertex: clean wrapper for reconnection functions
// =====================================================================================================================

struct SurfaceVertex {
    float3 x;           // world position
    float3 n_s;         // shading normal (world)
    float3 n_g;         // geometric normal (world)
    float3 o;           // outgoing/view direction (world, normalized)
    float3 Kd;          // albedo (refetched from texture)
    float  Pr;          // roughness
    float  Pm;          // metallic
    float  etai;        // IOR (incident side)
    float  etat;        // IOR (transmitted side)
    uint   matID;       // material ID
    float2 uv;          // texture coordinates
};

// Lightweight position-only reconstruction (~6 loads + 20 ALU)
inline float3 ReconstructPosition(uint instID, uint primID, float2 bary)
{
    const uint baseI = instanceProps[instID].indexBase;
    const uint i0 = indices[baseI + 3u * primID + 0u];
    const uint i1 = indices[baseI + 3u * primID + 1u];
    const uint i2 = indices[baseI + 3u * primID + 2u];
    const float3 p0 = BTriVertex[i0].vertex;
    const float3 p1 = BTriVertex[i1].vertex;
    const float3 p2 = BTriVertex[i2].vertex;
    const float b0 = 1.0f - bary.x - bary.y;
    float3 pLocal = p0 * b0 + p1 * bary.x + p2 * bary.y;
    return mul(instanceProps[instID].objectToWorld, float4(pLocal, 1.0f)).xyz;
}

// Full reconstruction via EvalSurfaceState + RefetchMaterial (~20 scattered loads)
inline SurfaceVertex BuildVertex(uint instID, uint primID, float2 bary, float3 viewOrigin)
{
    SurfaceVertex v;
    HitInfo h = EvalSurfaceState(instID, primID, bary, viewOrigin, 0);
    v.x     = h.hitPos;
    v.n_s   = h.hitNormal;
    v.n_g   = h.hitGNormal;
    v.o     = normalize(viewOrigin - h.hitPos);
    v.matID = GetMatIDFast(instID, primID);
    v.uv    = h.uv;
    RefetchMaterial(v.matID, h.uv, v.Kd, v.Pr, v.Pm);
    v.etai  = 0.0f;
    v.etat  = 0.0f;
    return v;
}

// Lightweight reconstruction using cached G-buffer data (~8 loads vs 20)
// Skips vertex normal/UV loads by using pre-cached values from the G-buffer
inline SurfaceVertex BuildVertexLight(
    uint instID, uint primID, float2 bary,
    float3 n1s_world, float3 n1g_world, float2 uv,
    float etai, float etat, float3 viewOrigin)
{
    SurfaceVertex v;
    v.x     = ReconstructPosition(instID, primID, bary);
    v.n_s   = n1s_world;
    v.n_g   = n1g_world;
    v.o     = normalize(viewOrigin - v.x);
    v.matID = GetMatIDFast(instID, primID);
    v.uv    = uv;
    RefetchMaterial(v.matID, uv, v.Kd, v.Pr, v.Pm);
    v.etai  = etai;
    v.etat  = etat;
    return v;
}

// v2 wrappers: clean API over old ReconnectDI/GI
inline float3 ReconnectDI_v2(SurfaceVertex v1, float3 x2, float3 n2, float3 L2, uint objID_di)
{
    return ReconnectDI(v1.x, v1.n_s, v1.n_g, v1.o, v1.matID,
                       x2, n2, L2, v1.Kd, v1.Pr, v1.Pm, v1.etai, v1.etat, objID_di);
}

inline float3 ReconnectGI_v2(
    SurfaceVertex v1, SurfaceVertex v2,
    float3 L2, float pdfx2_cached, float Jc,
    bool applyJ, out float Jn_out, out float J_out)
{
    return ReconnectGI(
        v1.x, v1.n_s, v1.n_g, v1.o, v1.matID,
        v1.Kd, v1.Pr, v1.Pm, v1.etai, v1.etat,
        v2.matID, v2.x, v2.n_s, v2.n_g, L2, v2.o,
        v2.Kd, v2.Pr, v2.Pm, v2.etai, v2.etat,
        pdfx2_cached, Jc, applyJ, Jn_out, J_out);
}

#endif