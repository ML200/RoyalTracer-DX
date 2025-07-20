#ifndef ENABLE_RAY_QUERY_INLINE
// The remaining functions remain unchanged.
float VisibilityCheck(
    float3 x1,
    float3 x2,
    float3 n1
)
{
    float V = 0.0f;
    float3 dir = x2-x1;
    float dist = length(dir);
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

    RayDesc ray;
    ray.Origin    = P + normalize(N) * EPSILON;            // ← offset *along ray*
    ray.Direction = dir;
    ray.TMin      = EPSILON;
    ray.TMax      = max(len - EPSILON*10.0f, 2.0f*EPSILON);

    RayQuery< RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH
              //|RAY_FLAG_CULL_BACK_FACING_TRIANGLES
              > rq;

    rq.TraceRayInline(SceneBVH,        // TLAS
                      RAY_FLAG_NONE,   // dynamic flags
                      0xFF,            // mask
                      ray);

    rq.Proceed();            // ← DRIVE TO COMPLETION

    return (rq.CommittedStatus() == COMMITTED_TRIANGLE_HIT) ? 0.0 : 1.0;
}

//------------------------------------------------------------------------------
//  fetch all per-triangle data and fill the four OUT parameters
//------------------------------------------------------------------------------
inline void EvalSurface(
    uint   instID,
    uint   primID,
    float2 bc2,                      // (u,v) from RayQuery
    out float3 outPosW,
    out float3 outNormW,
    out float  outArea,
    out uint   outMatID)
{
    uint  baseI = instanceProps[instID].indexBase;
    uint  baseV = instanceProps[instID].vertexBase;
    uint  baseM = instanceProps[instID].materialBase;

    // -- 1. vertex & index fetch -------------------------------------------
    uint idx0 = indices[baseI + 3u*primID + 0];
    uint idx1 = indices[baseI + 3u*primID + 1];
    uint idx2 = indices[baseI + 3u*primID + 2];

    float3 p0 = BTriVertex[idx0].vertex;
    float3 p1 = BTriVertex[idx1].vertex;
    float3 p2 = BTriVertex[idx2].vertex;

    // barycentrics = (1-u-v, u, v)
    float3 bary = float3(1.0 - bc2.x - bc2.y, bc2.x, bc2.y);

    // -- 2. hit position in object space ------------------------------------
    float3 posObj = p0*bary.x + p1*bary.y + p2*bary.z;

    // -- 3. flat geometric normal + area ------------------------------------
    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;
    float3 flatN  = normalize(cross(e1, e2));
    float  area_l = 0.5f * length(cross(e1, e2));

    // -- 4. smooth-shaded normal (with flat fallback) -----------------------
    float3 vn0 = BTriVertex[idx0].normal.xyz;
    float3 vn1 = BTriVertex[idx1].normal.xyz;
    float3 vn2 = BTriVertex[idx2].normal.xyz;

    float3 smoothN =flatN;
    //(all(vn0 != 0.0f) ? vn0 : flatN) * bary.x +
    //(all(vn1 != 0.0f) ? vn1 : flatN) * bary.y +
    //(all(vn2 != 0.0f) ? vn2 : flatN) * bary.z;

    float3 n = normalize(smoothN);

    // -- 5. object → world transforms ---------------------------------------
    outPosW  = mul(instanceProps[instID].objectToWorld,
                   float4(posObj,1)).xyz;

    n        = mul(instanceProps[instID].objectToWorldNormal,
                   float4(n,0)).xyz;
    outNormW = normalize(n);
    outArea  = area_l;

    // -- 6. material ID (same “normal.w” convention) ------------------------
    uint matOffset = asuint( BTriVertex[idx0].normal.w );   // stored in w
    outMatID       = materialIDs[baseM + 3u*primID + matOffset];
}


// Works in any shader stage that supports RayQuery (cs_6_6 / cs_6_7)
inline bool TraceRayInline_HitInfo(
    RaytracingAccelerationStructure SceneBVH,
    RayDesc                      ray,
    out HitInfo                  hit,
    uint                         rayFlags      = RAY_FLAG_NONE,
    uint                         instanceMask  = 0xFF)
{
    RayQuery< RAY_FLAG_NONE /* query template flags   */
             /* add SKIP_PROCEDURAL, CULL_NON_OPAQUE … here if desired */ > rq;

    rq.TraceRayInline(SceneBVH, rayFlags, instanceMask, ray);

    // Drive the query to completion
    while (rq.Proceed());

    if (rq.CommittedStatus() != COMMITTED_TRIANGLE_HIT)
        return false;                         // → miss; payload undefined

    // -------------------------------------------------------------------
    // Re‑create the CHS payload
    // -------------------------------------------------------------------
    float2 bc2    = rq.CommittedTriangleBarycentrics();  // (u,v)
    uint   primID = rq.CommittedPrimitiveIndex();
    uint   instID = rq.CommittedInstanceID();
    float  t      = rq.CommittedRayT();

    float3 surfPos, surfNormal;
    float  surfArea;
    uint   matID;
    EvalSurface(instID, primID, bc2,
                surfPos, surfNormal, surfArea, matID);

    float3 incoming = -ray.Direction;         // vector *towards* the camera
    if (dot(incoming, surfNormal) < 0.0f)     // < 0 → back‑face
        surfNormal = -surfNormal;


    hit.hitPosition = ray.Origin + t * ray.Direction;    // identical to CHS
    hit.hitNormal   = surfNormal;
    hit.area        = surfArea;
    hit.materialID  = matID;
    hit.objID       = instID;

    return true;     // hit filled
}
#endif