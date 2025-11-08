StructuredBuffer<uint> gTriToLightId : register(t15);


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


inline void EvalSurface(
    uint      instID,
    uint      primID,
    float2    bc2,
    float     t,
    in RayDesc ray,
    out HitInfo hit)
{
    const uint baseI = instanceProps[instID].indexBase;
    const uint baseM = instanceProps[instID].materialBase;

    // Indices
    const uint i0 = indices[baseI + 3u*primID + 0];
    const uint i1 = indices[baseI + 3u*primID + 1];
    const uint i2 = indices[baseI + 3u*primID + 2];

    // Vertices
    const float3 p0 = BTriVertex[i0].vertex;
    const float3 p1 = BTriVertex[i1].vertex;
    const float3 p2 = BTriVertex[i2].vertex;

    // Geometric normal + area in object space
    const float3 e1 = p1 - p0;
    const float3 e2 = p2 - p0;
    const float3 faceN_un = cross(e1, e2);
    const float  area_obj = 0.5f * length(faceN_un);
    const float3 flatN_obj = (area_obj > 0.0f) ? normalize(faceN_un) : float3(0,0,1);

    // Barycentrics
    const float3 bary = float3(1.0 - bc2.x - bc2.y, bc2.x, bc2.y);

    // Per-vertex shading normals
    float3 n0 = BTriVertex[i0].normal.xyz;
    float3 n1 = BTriVertex[i1].normal.xyz;
    float3 n2 = BTriVertex[i2].normal.xyz;


    float3 N_obj = flatN_obj;
    // Normalize inputs to be safe and clamp normal candidates to conservative values
    if (dot(n0,n0) >= EPSILON && dot(normalize(n0), flatN_obj) > 0.4f) n0 = normalize(n0); else n0 = flatN_obj;
    if (dot(n1,n1) >= EPSILON && dot(normalize(n1), flatN_obj) > 0.4f) n1 = normalize(n1); else n1 = flatN_obj;
    if (dot(n2,n2) >= EPSILON && dot(normalize(n2), flatN_obj) > 0.4f) n2 = normalize(n2); else n2 = flatN_obj;

    N_obj = normalize(n0*bary.x + n1*bary.y + n2*bary.z);

    // Adjust normal to always lie in the geometric normal plane
    if (dot(N_obj, flatN_obj) < 0.0f) {
        N_obj = normalize(N_obj - 2.0f * dot(N_obj, flatN_obj) * flatN_obj);
    }

    float4x4 normalMatrix = instanceProps[instID].objectToWorldNormal;

    // Transform geometric normal to world space for backface check
    float3 gNormW = mul(normalMatrix, float4(flatN_obj, 0)).xyz;

    // Transform shading normal to world space
    float3 nW = mul(normalMatrix, float4(N_obj, 0)).xyz;

    // Fill the HitInfo payload
    hit.hitPosition = ray.Origin + t * ray.Direction;
    hit.area        = area_obj;
    hit.materialID  = materialIDs[baseM + primID];
    hit.objID       = instID;
    hit.hitBackface = dot(ray.Direction, gNormW) > 0.0f;
    hit.hitNormal   = hit.hitBackface ? normalize(-nW) : normalize(nW);
    hit.hitGNormal   = hit.hitBackface ? normalize(-gNormW) : normalize(gNormW);

    {
        const float3 Vw   = normalize(-ray.Direction);
        const float  epsV = 0.1f;
        const float  epsR = 0.02f;
        hit.hitNormal = ClampNormalToViewAndReflection(hit.hitNormal, Vw, hit.hitGNormal, epsV, epsR);
    }

    // Light ID lookup
    const uint baseL = instanceProps[instID].triToLightBase;
    hit.lightID = gTriToLightId[baseL + primID];
}

inline void EvalMissHit(in RayDesc ray, out HitInfo hit)
{
    // Miss shader mirror
    hit.hitPosition = float3(0.4f, 0.4f, 0.5f); // for now blueish color
    hit.materialID  = 0xFFFFFFFFu; // material id is the env map

    float3 nd = (dot(ray.Direction, ray.Direction) > 1e-12f)
              ? normalize(ray.Direction)
              : float3(0.0f, 0.0f, 1.0f);

    hit.hitNormal   = -nd;              // face toward the camera
    hit.hitGNormal   = -nd;              // face toward the camera
    hit.hitBackface = false;            // not applicable for env
    hit.area        = 0.0f;             // infinite/none
    hit.objID       = 0xFFFFFFFFu;      // no instance
    hit.lightID     = 0xFFFFFFFFu;      // not a light
}

// INLINE BSDF ray, dont use for shadow ray because unoptimized
inline bool TraceRayInline_HitInfo(
    RaytracingAccelerationStructure SceneBVH,
    RayDesc ray,
    out HitInfo hit,
    uint rayFlags = RAY_FLAG_NONE,
    uint instanceMask = 0xFF)
{
    if(length(ray.Direction) < EPSILON) return false;
    RayQuery<RAY_FLAG_NONE> rq;
    rq.TraceRayInline(SceneBVH, rayFlags, instanceMask, ray);
    while (rq.Proceed());

    if (rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT){
        EvalMissHit(ray, hit);
        return false;
    }

    // Get hit details from the ray query
    const float2 bc2    = rq.CommittedTriangleBarycentrics();
    const uint   primID = rq.CommittedPrimitiveIndex();
    const uint   instID = rq.CommittedInstanceID();
    const float  t      = rq.CommittedRayT();

    // Evaluate the surface hit and write directly to the hit info struct
    EvalSurface(instID, primID, bc2, t, ray, hit);

    return true;
}
#endif