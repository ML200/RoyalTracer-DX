#ifdef ENABLE_RAY_QUERY_INLINE
// Sample a BSDF sample (generalized for solid angle)
BDReturn SampleBD_gen(
    in float3 x1,
    in float3 n1,
    in uint matID,
    in float3 o,
    inout uint waveSeed,
    inout uint2 threadSeed
){
    // Sample a light based on x1 and n1 using the light tree
    LT_Sample pick = LT_SampleLight(sdata.x1, sdata.n1, threadSeed.x);
    // Sample a point on the light x3
    uint triID = pick.id;
    uint   inst = g_EmissiveTriangles[triID].instanceID;
    float3 X = mul(instanceProps[inst].objectToWorld, float4(g_EmissiveTriangles[triID].x, 1)).xyz;
    float3 Y = mul(instanceProps[inst].objectToWorld, float4(g_EmissiveTriangles[triID].y, 1)).xyz;
    float3 Z = mul(instanceProps[inst].objectToWorld, float4(g_EmissiveTriangles[triID].z, 1)).xyz;

    float u = RandomFloatSingle(threadSeed.x);
    float v = RandomFloatSingle(threadSeed.x);
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    float3 x2 = (1.0f - u - v) * X + u * Y + v * Z;

    float3 e1 = Y - X, e2 = Z - X;
    float3 n  = cross(e1, e2);
    float  A  = 0.5f * length(n);
    float3 nL = n / max(2.0f*A, EPSILON);

    // Sample a direction in the cosine hemisphere (1 ray)
    float3 dir = RandomUnitVectorInHemisphere(n, threadSeed);

    // Trace the direction to get x2
    RayDesc ray;
    ray.Origin = x1;
    ray.Direction = dir;
    ray.TMin = 0.001f;
    ray.TMax = 10000.0f;
    HitInfo samplePayload;
    TraceRayInline_HitInfo(SceneBVH, ray, samplePayload, RAY_FLAG_NONE, 0xFF);
    if(any(materials[samplePayload.materialID].Ke > 0.0f)) return (BDReturn)0.0f; // Early out because of termination on light

    // Reconnect x2 to x1 (Check that its not occluded and also in front of the surface); (1 ray)
    float V = VisibilityCheckCP(x1, x2, n1);
    if(V == 0.0f) return (BDReturn)0.0f;

    // Compute the joined pdf

    //return the sample. We evaluate it using the extended reconnection
    BDReturn sreturn = (BDReturn)0;
    sreturn.x2 = samplePayload.hitPosition;
    sreturn.n2 = samplePayload.hitNormal;
    sreturn.L2 = g_EmissiveTriangles[triID].emission;
    sreturn.objID = samplePayload.objID;
    sreturn.matID = samplePayload.materialID;

    sreturn.pdf = pdf_b;

    return sreturn;
}
#endif
