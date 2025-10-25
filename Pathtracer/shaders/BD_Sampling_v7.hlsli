#ifdef ENABLE_RAY_QUERY_INLINE
// Sample a Bidirectional sample
BDReturn SampleForward(
    in float3 x1,
    in float3 n1,
    in uint matID,
    in float3 o,
    inout uint waveSeed,
    inout uint2 threadSeed
){
    // Sample a light based on x1 and n1 using the light tree
    LT_Sample pick = LT_SampleLight_Indirect(x1, n1, threadSeed.x);
    // Sample a point on the light x3
    uint triID = pick.id;
    uint   inst = g_EmissiveTriangles[triID].instanceID;
    float3 X = mul(instanceProps[inst].objectToWorld, float4(g_EmissiveTriangles[triID].x, 1)).xyz;
    float3 Y = mul(instanceProps[inst].objectToWorld, float4(g_EmissiveTriangles[triID].y, 1)).xyz;
    float3 Z = mul(instanceProps[inst].objectToWorld, float4(g_EmissiveTriangles[triID].z, 1)).xyz;

    float u = RandomFloatSingle(threadSeed.x);
    float v = RandomFloatSingle(threadSeed.x);
    if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
    float3 xl = (1.0f - u - v) * X + u * Y + v * Z;

    float3 e1 = Y - X, e2 = Z - X;
    float3 n  = cross(e1, e2);
    float  A  = 0.5f * length(n);
    float3 nL = n / max(2.0f*A, EPSILON);

    // Sample a direction in the cosine hemisphere (1 ray)
    float3 dir = normalize(RandomUnitVectorInHemisphere(nL, threadSeed));

    // Trace the direction to get xl
    RayDesc ray;
    ray.Origin = xl;
    ray.Direction = dir;
    ray.TMin = 0.001f;
    ray.TMax = 10000.0f;
    HitInfo samplePayload;
    TraceRayInline_HitInfo(SceneBVH, ray, samplePayload, RAY_FLAG_NONE, 0xFF);
    if(any(materials[samplePayload.materialID].Ke > 0.0f)) return (BDReturn)0.0f; // Early out because of termination on light

    // Reconnect x2 to x1 (Check that its not occluded and also in front of the surface); (1 ray)
    float V = VisibilityCheckCP(x1, samplePayload.hitPosition, n1);
    if(V == 0.0f) return (BDReturn)0.0f;

    // Compute the joined pdf
    float3 d32   = samplePayload.hitPosition - xl;
    float  r32_2 = max(dot(d32, d32), 1e-20f);
    float3 w32   = normalize(d32);
    float  cosL  = max(dot(nL, dir), 0.0f);
    float  cos2_32 = max(dot(samplePayload.hitNormal, -w32), 0.0f);

    float pdf_area_x2 = pick.pdf
                      * (1.0f / max(A, 1e-20f))
                      * (cosL / PI);

    // convert AREA@x2 -> SOLID-ANGLE@x1
    float3 d12   = samplePayload.hitPosition - x1;
    float  r12_2 = max(dot(d12, d12), 1e-20f);
    float3 w12   = normalize(d12);
    float  cos2_12 = max(dot(samplePayload.hitNormal, -w12), 0.0f);

    float pdf_omega_x1 = pdf_area_x2 * (r12_2 / max(cos2_12, 1e-20f));

    //return the sample. We evaluate it using the extended reconnection
    BDReturn sreturn = (BDReturn)0;
    sreturn.x2 = samplePayload.hitPosition;
    sreturn.n2 = samplePayload.hitNormal;
    sreturn.L2 = g_EmissiveTriangles[triID].emission;
    sreturn.objID = samplePayload.objID;
    sreturn.matID = samplePayload.materialID;
    sreturn.x3 = xl;
    sreturn.n3 = nL;

    sreturn.pdf = pdf_omega_x1;

    return sreturn;
}
#endif


#ifdef ENABLE_RAY_QUERY_INLINE
// Sample a Bidirectional sample
BDReturn SampleBackwardNEE(
    in float3 x1,
    in float3 n1,
    in uint matID,
    in float3 o,
    inout uint waveSeed,
    inout uint2 threadSeed
){
    //Sample a bsdf ray and eval its pdf
    float3 sampleBSDF;
    uint strategy = SelectSamplingStrategy(matID, o, n1, threadSeed);
    SampleBRDF(strategy, matID, o, n1, n1, sampleBSDF, x1, threadSeed);
    if(all(sampleBSDF == 0.0f)) return (BDReturn)0;

    RayDesc ray;
    ray.Origin = x1;
    ray.Direction = sampleBSDF;
    ray.TMin = 0.001f;
    ray.TMax = 10000.0f;
    HitInfo samplePayload;
    TraceRayInline_HitInfo(SceneBVH, ray, samplePayload, RAY_FLAG_NONE, 0xFF);
    if(any(materials[samplePayload.materialID].Ke > 0.0f)) return (BDReturn)0;
    pdfBSDF = BRDF_PDF_COMBINED(matID, n1, -sampleBSDF, o);

    // Some intermediate variables:
    float3 x2 = samplePayload.hitPosition;
    float3 n2 = samplePayload.hitNormal;
    uint matID2 = samplePayload.materialID;
    float3 o2 = x1 - x2;


    // Sample an NEE ray and eval its pdf


    //return the sample. We evaluate it using the extended reconnection
    BDReturn sreturn = (BDReturn)0;
    sreturn.x2 = samplePayload.hitPosition;
    sreturn.n2 = samplePayload.hitNormal;
    sreturn.L2 = g_EmissiveTriangles[triID].emission;
    sreturn.objID = samplePayload.objID;
    sreturn.matID = samplePayload.materialID;
    sreturn.x3 = xl;
    sreturn.n3 = nL;

    sreturn.pdf = pdf_omega_x1;

    return sreturn;
}
#endif
