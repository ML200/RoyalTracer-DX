// Minimal payload type (must be a user-defined struct)
struct TracePayload
{
    uint dummy;
};

// Optional: if you prefer your own wrapper instead of BuiltInTriangleIntersectionAttributes
struct TriAttrs
{
    float2 bary; // numeric-only, OK
};


#ifndef ENABLE_RAY_QUERY_INLINE
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
#endif

#ifdef ENABLE_RAY_QUERY_INLINE
float VisibilityCheckCP(float3 P, float3 L, float3 N)
{
    float3 dir = L - P;
    if(length(dir)<EPSILON) return 0.0f;
    dir = normalize(dir);
    float  len = length(L - P);

    if(dot(dir, N) < 0.0f)
            N = -N;

    RayDesc ray;
    ray.Origin    = P + normalize(N) * SBIAS * 0.5f;
    ray.Direction = dir;
    ray.TMin      = EPSILON;
    ray.TMax      = max(len - SBIAS * 10.0f - EPSILON * 10.0f, 2.0f * EPSILON);

    RayQuery< RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
              //|RAY_FLAG_CULL_BACK_FACING_TRIANGLES
              > rq;

    rq.TraceRayInline(SceneBVH,        // TLAS
                      RAY_FLAG_NONE,   // dynamic flags
                      0xFF,            // mask
                      ray);

    rq.Proceed();

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
        albedo = albedoTextures.SampleLevel(g_sampler, float3(albedoUV, mat.albedoTexID), level).rgb;
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
        float4 rmaSample = rmaTextures.SampleLevel(g_sampler, float3(rmaUV, mat.rmaTexID), level);

        pbrProps.x = rmaSample.g; // Roughness
        pbrProps.y = rmaSample.b; // Metallic
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
    dx::HitObject hitObj = dx::HitObject::TraceRay(SceneBVH, rayFlags, instanceMask, 0, 1, 0, ray, payload);

    uint hint = hitObj.IsHit()?1:0;
    dx::MaybeReorderThread(hitObj, hint, 1);
    return hitObj;
}


inline float3 EvalMissState()
{
    // Miss shader
    return float3(0.8f, 0.8f, 0.8f);
}

inline HitInfo EvalSurfaceState(
    uint      instID,
    uint      primID,
    float2    bc2,
    float3    origin,
    uint level
    )
{
    // 1. DATA GATHER
    const uint baseI = instanceProps[instID].indexBase;
    const uint baseM = instanceProps[instID].materialBase;
    const uint materialID = materialIDs[baseM + primID];

    const uint i0 = indices[baseI + 3u*primID + 0];
    const uint i1 = indices[baseI + 3u*primID + 1];
    const uint i2 = indices[baseI + 3u*primID + 2];

    const STriVertex v0 = BTriVertex[i0];
    const STriVertex v1 = BTriVertex[i1];
    const STriVertex v2 = BTriVertex[i2];

    // 2. LOCAL GEOMETRY
    const float3 bary = float3(1.0f - bc2.x - bc2.y, bc2.x, bc2.y);

    const float3 p_local = v0.vertex * bary.x + v1.vertex * bary.y + v2.vertex * bary.z;

    const float2 uv0 = (float2)v0.texCoord;
    const float2 uv1 = (float2)v1.texCoord;
    const float2 uv2 = (float2)v2.texCoord;
    const float2 uv  = uv0 * bary.x + uv1 * bary.y + uv2 * bary.z;

    float3 n0 = UnpackNormal_INT(v0.packedNormal);
    float3 n1 = UnpackNormal_INT(v1.packedNormal);
    float3 n2 = UnpackNormal_INT(v2.packedNormal);

    // Fix degenerate normals logic (kept same as yours)
    const float3 e1_local = v1.vertex - v0.vertex;
    const float3 e2_local = v2.vertex - v0.vertex;
    const float3 faceN_local_un = cross(e1_local, e2_local);
    const float  area_obj       = 0.5f * length(faceN_local_un);
    const float3 flatN_obj      = (area_obj > 0.0f) ? normalize(faceN_local_un) : float3(0,0,1);

    if (dot(n0, n0) < EPSILON || dot(normalize(n0), flatN_obj) <= 0.4f) n0 = flatN_obj;
    if (dot(n1, n1) < EPSILON || dot(normalize(n1), flatN_obj) <= 0.4f) n1 = flatN_obj;
    if (dot(n2, n2) < EPSILON || dot(normalize(n2), flatN_obj) <= 0.4f) n2 = flatN_obj;

    float3 n_local = normalize(n0 * bary.x + n1 * bary.y + n2 * bary.z);

    if (dot(n_local, flatN_obj) < 0.0f) {
        n_local = normalize(n_local - 2.0f * dot(n_local, flatN_obj) * flatN_obj);
    }

    // 3. TRANSFORM
    float3 posW, normW, geoNormW, tangentW_geom;

    {
        float4x4 M = instanceProps[instID].objectToWorld;
        float3x3 M_rot = (float3x3)M;

        posW = mul(M, float4(p_local, 1.0f)).xyz;
        normW    = normalize(mul(M_rot, n_local));
        geoNormW = normalize(mul(M_rot, flatN_obj));

        // Calculate Tangent in World Space using transformed vertices
        // We do this here because non-uniform scaling might skew local tangents
        float3 p0W = mul(M, float4(v0.vertex, 1.0)).xyz;
        float3 p1W = mul(M, float4(v1.vertex, 1.0)).xyz;
        float3 p2W = mul(M, float4(v2.vertex, 1.0)).xyz;

        tangentW_geom = CalculateGeometricTangent(p0W, p1W, p2W, uv0, uv1, uv2, normW);
    }

    // 4. MATERIAL & TEXTURING
    const Material mat = materials[materialID];
    HitInfo hit = (HitInfo)0.0f;
    hit.localKd = EvaluateAlbedo(mat, uv, level);
    float2 pbr  = EvaluatePBRProperties(mat, uv, level);
    hit.localPr = pbr.x;
    hit.localPm = pbr.y;

    // 5. NORMAL MAPPING (FIXED)
    if (mat.normalTexID != -1)
    {
        float2 normalUV = uv * mat.normalUVScale;

        // Use the geometric tangent we calculated above
        float3 T = tangentW_geom;

        // Re-orthogonalize T with respect to the shading normal
        float3 tangentW   = normalize(T - dot(T, normW) * normW);
        // Compute Bitangent
        float3 bitangentW = cross(normW, tangentW);

        float3x3 tbn = float3x3(tangentW, bitangentW, normW);

        float3 n_tangent = normalTextures.SampleLevel(g_sampler, float3(normalUV, mat.normalTexID), level).xyz * 2.0f - 1.0f;

        normW = normalize(mul(n_tangent, tbn));
    }

    // 6. FINALIZE HIT INFO
    float3 viewDir = normalize(posW - origin);

    hit.hitT = length(posW - origin);
    hit.materialID  = materialID;
    hit.objID       = instID;
    const bool isBackface = (dot(viewDir, geoNormW) > 0.0f);
    hit.hitNormal   = isBackface ? -normW : normW;
    hit.hitGNormal  = isBackface ? -geoNormW : geoNormW;

    {
        const float3 Vw = -viewDir;
        hit.hitNormal = ClampNormalToViewAndReflection(hit.hitNormal, Vw, hit.hitGNormal, 0.1f, 0.02f);
    }

    const uint baseL = instanceProps[instID].triToLightBase;
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