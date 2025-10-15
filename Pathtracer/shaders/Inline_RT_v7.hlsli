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
    float3 dir = normalize(L - P);
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

// Inline version of closest hit shader
inline void EvalSurface(
    uint   instID,
    uint   primID,
    float2 bc2,
    out float3 outPosW,
    out float3 outNormW,
    out float  outArea,
    out uint   outMatID)
{
    uint  baseI = instanceProps[instID].indexBase;
    uint  baseV = instanceProps[instID].vertexBase;
    uint  baseM = instanceProps[instID].materialBase;

    // vertex data
    uint idx0 = indices[baseI + 3u*primID + 0];
    uint idx1 = indices[baseI + 3u*primID + 1];
    uint idx2 = indices[baseI + 3u*primID + 2];

    float3 p0 = BTriVertex[idx0].vertex;
    float3 p1 = BTriVertex[idx1].vertex;
    float3 p2 = BTriVertex[idx2].vertex;

    // barycentrics = (1-u-v, u, v)
    float3 bary = float3(1.0 - bc2.x - bc2.y, bc2.x, bc2.y);

    // position
    float3 posObj = p0*bary.x + p1*bary.y + p2*bary.z;

    // area + norm
    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;
    float3 flatN  = normalize(cross(e1, e2));
    float  area_l = 0.5f * length(cross(e1, e2));

    // TODO: smooth normal is borked, fix needed
    float3 vn0 = BTriVertex[idx0].normal.xyz;
    float3 vn1 = BTriVertex[idx1].normal.xyz;
    float3 vn2 = BTriVertex[idx2].normal.xyz;

    float3 smoothN = flatN;
    /*(length(vn0) != 0.0f ? vn0 : flatN) * bary.x +
    (length(vn1) != 0.0f ? vn1 : flatN) * bary.y +
    (length(vn2) != 0.0f ? vn2 : flatN) * bary.z;*/

    float3 n = normalize(smoothN);

    // object to world
    outPosW  = mul(instanceProps[instID].objectToWorld,
                   float4(posObj,1)).xyz;

    n        = mul(instanceProps[instID].objectToWorldNormal,
                   float4(n,0)).xyz;
    outNormW = normalize(n);
    outArea  = area_l;

    // material id
    uint matOffset = asuint( BTriVertex[idx0].normal.w );
    outMatID       = materialIDs[baseM + 3u*primID + matOffset];
}

// INLINE BSDF ray, dont use for shadow ray because unoptimized
inline bool TraceRayInline_HitInfo(
    RaytracingAccelerationStructure SceneBVH,
    RayDesc ray,
    out HitInfo hit,
    uint rayFlags = RAY_FLAG_NONE,
    uint instanceMask = 0xFF)
{
    RayQuery<RAY_FLAG_NONE> rq;
    rq.TraceRayInline(SceneBVH, rayFlags, instanceMask, ray);
    while (rq.Proceed());

    if (rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return false;

    float2 bc2    = rq.CommittedTriangleBarycentrics();
    uint   primID = rq.CommittedPrimitiveIndex();
    uint   instID = rq.CommittedInstanceID();
    float  t      = rq.CommittedRayT();

    float3 surfPos, surfNormal;
    float  surfArea;
    uint   matID;
    EvalSurface(instID, primID, bc2, surfPos, surfNormal, surfArea, matID);

    float3 incoming = -ray.Direction;
    //if (dot(incoming, surfNormal) < 0.0f) surfNormal = -surfNormal;

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