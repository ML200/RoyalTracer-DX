// Minimal payload type
struct [raypayload] TracePayload
{
    uint dummy : read(caller) : write(caller);
};

// Optional: if you prefer your own wrapper instead of BuiltInTriangleIntersectionAttributes
struct TriAttrs
{
    float2 bary; // numeric-only, OK
};


/*#ifndef ENABLE_RAY_QUERY_INLINE
float VisibilityCheck(
    float3 x1,
    float3 x2,
    float3 n1
)
{
    float V = 0.0f;
    float3 dir = x2-x1;
    float dist = length(dir);
    // Adjust normal offset based on ray direction (transmission reconnection requires negative offset)
    if(dot(dir, n1) < 0.0f)
        n1 = -n1;

    RayDesc ray;
    ray.Origin = x1 + normalize(n1) * EPSILON;
    ray.Direction = normalize(dir);
    ray.TMin = EPSILON;
    ray.TMax = max(dist - 10.0f * EPSILON, 2.0f * EPSILON);
    ShadowHitInfo shadowPayload;
    shadowPayload.isHit = false;
    const uint flags =
        RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    TraceRay(SceneBVH, flags, 0xFF, 1, 0, 1, ray, shadowPayload);
    V = shadowPayload.isHit ? 0.0f : 1.0f;
    return V;
}
#endif*/

#ifdef ENABLE_RAY_QUERY_INLINE
float VisibilityCheckCP(float3 P, float3 L, float3 N, uint objID)
{
    float3 dir = L - P;
    if(objID == 0xFFFFFFFFu || objID == 0xFFFFFFFEu) dir = normalize(L);
    if(length(dir)<EPSILON) return 0.0f;
    dir = normalize(dir);
    float  len = length(L - P);
    if(objID == 0xFFFFFFFFu || objID == 0xFFFFFFFEu) len = 10000.0f;

    if(dot(dir, N) < 0.0f)
            N = -N;

    RayDesc ray;
    ray.Origin    = P + normalize(N) * SBIAS * 0.5f;
    ray.Direction = dir;
    ray.TMin      = EPSILON * 2.0f;
    ray.TMax      = max(len - SBIAS * 10.0f - EPSILON * 10.0f, 2.0f * EPSILON);

    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> rq;

    rq.TraceRayInline(SceneBVH, RAY_FLAG_NONE, 0xFF, ray);

    while (rq.Proceed())
    {
        if (rq.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE)
        {
            uint instanceID  = rq.CandidateInstanceID();
            uint primitiveID = rq.CandidatePrimitiveIndex();
            float2 bary      = rq.CandidateTriangleBarycentrics();

            InstanceProperties inst = instanceProps[instanceID];

            uint primID = inst.opaqueTriCount + primitiveID;
            uint baseI  = inst.indexBase;

            uint i0 = indices[baseI + 3u * primID + 0u];
            uint i1 = indices[baseI + 3u * primID + 1u];
            uint i2 = indices[baseI + 3u * primID + 2u];

            float2 uv0 = (float2)BTriVertex[i0].texCoord;
            float2 uv1 = (float2)BTriVertex[i1].texCoord;
            float2 uv2 = (float2)BTriVertex[i2].texCoord;

            float2 uv = uv0 * (1.0 - bary.x - bary.y)
                      + uv1 * bary.x
                      + uv2 * bary.y;

            uint matID = materialIDs[inst.materialBase + primID];
            Material mat = materials[matID];

            float alpha = mat.Kd.w;
            if (mat.albedoTexID >= 0)
            {
                Texture2D<float4> tex = ResourceDescriptorHeap[mat.albedoTexID];
                alpha = tex.SampleLevel(g_sampler, uv, 0).a;
            }

            if (alpha < mat.alphaThreshold)
            continue;
            rq.CommitNonOpaqueTriangleHit();
        }
    }

    return (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0 : 1.0;
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

// Helpers for the surface eval
// =====================================================================================================================
void GetOrthonormalBasis(float3 N, out float3 T, out float3 B)
{
    float sign = (N.z >= 0.0) ? 1.0 : -1.0;
    const float a = -1.0 / (sign + N.z);
    const float b = N.x * N.y * a;
    T = float3(1.0 + sign * N.x * N.x * a, sign * b, -sign * N.x);
    B = float3(b, sign + N.y * N.y * a, -N.y);
}

float3 CalculateGeometricTangent(float3 p0, float3 p1, float3 p2, float2 uv0, float2 uv1, float2 uv2, float3 normal)
{
    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;
    float2 duv1 = uv1 - uv0;
    float2 duv2 = uv2 - uv0;

    float det = duv1.x * duv2.y - duv1.y * duv2.x;

    float3 T;
    if (abs(det) < 1e-6f)
    {
        // Fallback if UVs are degenerate
        float3 B_unused;
        GetOrthonormalBasis(normal, T, B_unused);
    }
    else
    {
        float invDet = 1.0f / det;
        T = normalize((e1 * duv2.y - e2 * duv1.y) * invDet);
    }

    return T;
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
// =====================================================================================================================

inline dx::HitObject TraceRay_Custom(
    RaytracingAccelerationStructure SceneBVH,
    RayDesc ray,
    uint rayFlags = RAY_FLAG_NONE,
    uint instanceMask = 0xFF)
{

    TracePayload payload = (TracePayload)0; // Dummy payload
    dx::HitObject hitObj = dx::HitObject::TraceRay(SceneBVH, rayFlags, instanceMask, 0, 0, 0, ray, payload);

    uint hint = hitObj.IsHit()?1:0;
    dx::MaybeReorderThread(hitObj, hint, 1);
    return hitObj;
}


inline float3 EvalMissState(float3 rayDir, float3 sunDisk)
{
    return EvaluateSky(rayDir);
    /*float3 sky = EvaluateSky(rayDir);

    SunState sun = ComputeSunState();

    CloudResult clouds = EvaluateClouds(
        rayDir,
        sun.dirWS,
        sun.tint * SUN_INTENSITY_VAL,
        EvaluateSky(float3(0, 1, 0)),
        EvaluateSky(float3(0, 0.1f, 1)) * 0.5f
    );

    return clouds.color + (sky + sunDisk) * clouds.transmit;*/
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
    hit.localKd = EvaluateAlbedo(mat, uv, level);

    {
        const float2 pbr = EvaluatePBRProperties(mat, uv, level);
        hit.localPr = (half)pbr.x;
        hit.localPm = (half)pbr.y;
    }

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

#endif