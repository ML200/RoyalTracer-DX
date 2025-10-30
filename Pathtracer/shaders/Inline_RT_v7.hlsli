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

inline void EvalSurface(
    uint   instID,
    uint   primID,
    float2 bc2,
    float3 rayDir,
    out float3 outPosW,
    out float3 outNormW,
    out float3 outGNormW,
    out float  outArea,
    out uint   outMatID)
{
    const uint baseI = instanceProps[instID].indexBase;
    const uint baseM = instanceProps[instID].materialBase;

    // indices
    const uint i0 = indices[baseI + 3u*primID + 0];
    const uint i1 = indices[baseI + 3u*primID + 1];
    const uint i2 = indices[baseI + 3u*primID + 2];

    // vertices
    const float3 p0 = BTriVertex[i0].vertex;
    const float3 p1 = BTriVertex[i1].vertex;
    const float3 p2 = BTriVertex[i2].vertex;

    const float3 bary = float3(1.0 - bc2.x - bc2.y, bc2.x, bc2.y);

    // position
    const float3 posObj = p0*bary.x + p1*bary.y + p2*bary.z;

    // geometric normal + area
    const float3 e1 = p1 - p0;
    const float3 e2 = p2 - p0;
    const float3 faceN_un = cross(e1, e2);
    const float  area_l   = 0.5f * length(faceN_un);
    const float3 flatN    = (area_l > 0.0f) ? normalize(faceN_un) : float3(0,0,1);

    // per-vertex normals from buffer
    float3 n0 = BTriVertex[i0].normal.xyz;
    float3 n1 = BTriVertex[i1].normal.xyz;
    float3 n2 = BTriVertex[i2].normal.xyz;

    // decide: all-zero => flat; otherwise interpolate
    const float eps = 1e-10;
    const bool allZero = (dot(n0,n0) < eps) && (dot(n1,n1) < eps) && (dot(n2,n2) < eps);

    float3 N = flatN;
    if (!allZero) {
        // normalize inputs to be safe (exporters may give near-unit)
        if (dot(n0,n0) >= eps) n0 = normalize(n0); else n0 = flatN;
        if (dot(n1,n1) >= eps) n1 = normalize(n1); else n1 = flatN;
        if (dot(n2,n2) >= eps) n2 = normalize(n2); else n2 = flatN;

        N = normalize(n0*bary.x + n1*bary.y + n2*bary.z);

        // keep same hemisphere as the face normal to avoid flips
        if (dot(N, flatN) < 0.0f) N = flatN;
    }

    // object -> world
    outPosW  = mul(instanceProps[instID].objectToWorld, float4(posObj,1)).xyz;
    float3 nW = mul(instanceProps[instID].objectToWorldNormal, float4(N,0)).xyz;
    outNormW  = normalize(nW);
    outArea   = area_l;
    outMatID = materialIDs[baseM + primID];
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

    float2 bc2    = rq.CommittedTriangleBarycentrics();
    uint   primID = rq.CommittedPrimitiveIndex();
    uint   instID = rq.CommittedInstanceID();
    float  t      = rq.CommittedRayT();

    float3 surfPos, surfNormal, surfGNormal;
    float  surfArea;
    uint   matID;
    EvalSurface(instID, primID, bc2, ray.Direction, surfPos, surfNormal, surfGNormal, surfArea, matID);

    hit.hitBackface = dot(ray.Direction, surfGNormal) > 0.0f;
    hit.hitPosition = ray.Origin + t * ray.Direction;
    hit.hitNormal   = surfNormal;
    hit.area        = surfArea;
    hit.materialID  = matID;
    hit.objID       = instID;

    uint base   = instanceProps[instID].triToLightBase;
    uint light  = gTriToLightId[base + primID];
    hit.lightID = light;

    return true;
}
#endif