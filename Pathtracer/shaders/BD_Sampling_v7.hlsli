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
    sreturn.triID = triID;

    sreturn.pdf = pdf_omega_x1;
    sreturn.pdf_seg = pdf_omega_x1;

    return sreturn;
}
#endif


inline float PdfFromForward(
    float3 x1,
    BDReturn bd
){
    const float3 x2 = bd.x2;
    const float3 n2 = bd.n2;
    const float3 x3 = bd.x3;
    const float3 n3 = bd.n3;
    const uint triID = bd.triID;

    // 1) Area pdf at the emitter point x3 (triangle pick × uniform point on that triangle)
    const uint  lightObjID  = g_EmissiveTriangles[triID].instanceID;
    const float pA_x3       = LT_Pdf_LightTree_Area_Indirect(x1, float3(0,0,0), triID, lightObjID);
    if (pA_x3 <= 0.0f) return 0.0f;

    // 2) Direction pdf at the light
    const float3 d32  = x2 - x3;
    const float  r32_2= max(dot(d32, d32), EPSILON);
    const float3 w32  = d32 * rsqrt(r32_2);
    const float  cos3 = max(dot(n3, w32), 0.0f);
    const float  p_dir = (cos3 / PI);
    if (p_dir <= 0.0f) return 0.0f;

    // 3) Convert A@x2 -> Ω@x1
    const float3 d12  = x2 - x1;
    const float  r12_2= max(dot(d12, d12), EPSILON);
    const float3 w12  = d12 * rsqrt(r12_2);
    const float  cos2_12 = max(dot(n2, -w12), 0.0f);
    if (cos2_12 <= 0.0f) return 0.0f;

    const float pdf = (pA_x3 * p_dir) * (r12_2 / max(cos2_12, EPSILON));
    return pdf;
}



#ifdef ENABLE_RAY_QUERY_INLINE
// Sample a BSDF + NEE sample
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
    float pdfBSDF = BRDF_PDF_COMBINED(matID, n1, -sampleBSDF, o);

    // Some intermediate variables:
    float3 x2 = samplePayload.hitPosition;
    float3 n2 = samplePayload.hitNormal;
    uint matID2 = samplePayload.materialID;
    float3 o2 = x1 - x2;


    // Sample an NEE ray and eval its pdf
    // Sample a light and a point on it
    LT_Sample pick = LT_SampleLight(x2, n2, threadSeed.x);
    uint lightIdx = pick.id;
    float xi1 = RandomFloatSingle(threadSeed.x);
    float xi2 = RandomFloatSingle(threadSeed.x);
    if (xi1 + xi2 > 1.0f) {
        xi1 = 1.0f - xi1;
        xi2 = 1.0f - xi2;
    }
    float uu = 1.0f - xi1 - xi2;
    float vv = xi1;
    float ww = xi2;
    float3 x = mul(instanceProps[g_EmissiveTriangles[lightIdx].instanceID].objectToWorld, float4(g_EmissiveTriangles[lightIdx].x, 1.f)).xyz;
    float3 y = mul(instanceProps[g_EmissiveTriangles[lightIdx].instanceID].objectToWorld, float4(g_EmissiveTriangles[lightIdx].y, 1.f)).xyz;
    float3 z = mul(instanceProps[g_EmissiveTriangles[lightIdx].instanceID].objectToWorld, float4(g_EmissiveTriangles[lightIdx].z, 1.f)).xyz;
    float3 x3 = uu * x + vv * y + ww * z;

    float V = VisibilityCheckCP(x2, x3, n2);
    if(V == 0.0f) return (BDReturn)0.0f;

    // Compute sample direction and normalized vector
    float3 L = x3 - x2;
    float dist2 = dot(L, L);
    float dist = sqrt(dist2);
    float3 L_norm = L / dist;

    // Compute normal and area
    float3 edge1 = y - x;
    float3 edge2 = z - x;
    float3 normalL = cross(edge1, edge2);
    float area_L = 0.5f * length(normalL);
    float3 n3 = normalL / (2.0f * area_L + EPSILON);

    if (dot(n3, -L_norm) < EPSILON) {
        return (BDReturn)0;
    }

    // Compute NEE PDF (solid angle!)
    float pdfNEE = pick.pdf / max(area_L, 1e-10f) * dist2 / max(dot(n3, -L_norm), 0.0f);


    //return the sample. We evaluate it using the extended reconnection
    BDReturn sreturn = (BDReturn)0;
    sreturn.x2 = x2;
    sreturn.n2 = n2;
    sreturn.L2 = g_EmissiveTriangles[lightIdx].emission;
    sreturn.objID = samplePayload.objID;
    sreturn.matID = matID2;
    sreturn.x3 = x3;
    sreturn.n3 = n3;
    sreturn.triID = lightIdx;

    sreturn.pdf = pdfBSDF * pdfNEE;
    sreturn.pdf_seg = pdfBSDF;

    return sreturn;
}
#endif



#ifdef ENABLE_RAY_QUERY_INLINE
// Sample a BSDF + NEE sample
BDReturn SampleSingleBSDF(
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
    float pdfBSDF = BRDF_PDF_COMBINED(matID, n1, -sampleBSDF, o);

    float3 tp = BSDF_term(matID, n1, -sampleBSDF, o) * max(dot(n1, sampleBSDF), 0.0f);

    //return the sample. We evaluate it using the extended reconnection
    BDReturn sreturn = (BDReturn)0;
    sreturn.x2 = samplePayload.hitPosition;
    sreturn.n2 = samplePayload.hitNormal;
    sreturn.L2 = tp;
    sreturn.objID = samplePayload.objID;
    sreturn.matID = samplePayload.materialID;
    sreturn.pdf = pdfBSDF;
    sreturn.pdf_seg = pdfBSDF;

    return sreturn;
}
#endif

inline float PdfFromBackwardNEE(
    float3 x1,
    float3 n1,
    uint   matID1,
    float3 o1,
    BDReturn bd
){
    //1) BSDF pdf at x1 in solid angle (toward x2)
    const float3 d12   = bd.x2 - x1;
    const float  r12_2 = max(dot(d12, d12), EPSILON);
    const float3 w12   = d12 * rsqrt(r12_2);
    const float  pdf_bsdf = BRDF_PDF_COMBINED(matID1, n1, -w12, o1);
    if (pdf_bsdf <= 0.0f) return 0.0f;

    // 2) NEE pdf from x2 to the chosen point x3
    const uint  lightObjID = g_EmissiveTriangles[bd.triID].instanceID;
    const float pA_x3      = LT_Pdf_LightTree_Area(bd.x2, bd.n2, bd.triID, lightObjID);
    if (pA_x3 <= 0.0f) return 0.0f;

    // Convert area -> solid angle
    const float3 d23   = bd.x3 - bd.x2;
    const float  r23_2 = max(dot(d23, d23), EPSILON);
    const float3 w23   = d23 * rsqrt(r23_2);
    const float  cos3  = max(dot(bd.n3, -w23), 0.0f);   // emitter facing x2
    if (cos3 <= 0.0f) return 0.0f;

    const float pdf_nee = pA_x3 * (r23_2 / cos3);
    return pdf_bsdf * pdf_nee;
}

// HELPERS
//-----------------------------------------------------------------------------------------------
inline float2 BD_MethodProbsFromRoughness(float roughness)
{
    float r = saturate(roughness);
    float t = saturate((r - 0.5f) * 2.0f);   // 0 at 0.5, 1 at 1.0
    float p_bsdf = 1.0f - 0.5f * t;          // 1.0 -> 0.5
    float p_fwd  = 1.0f - p_bsdf;            // 0.0 -> 0.5
    return float2(p_bsdf, p_fwd);
}

inline uint BD_PickMethod(float2 probs, inout uint2 threadSeed)
{
    float xi = RandomFloatSingle(threadSeed.x);
    return (xi < probs.x) ? 0u : 1u;
}

// Wrapper for sampling the selected technique
inline BDReturn BD_SamplePicked(
    in float3 x1,
    in float3 n1,
    in uint matID,
    in float3 o,
    inout uint waveSeed,
    inout uint2 threadSeed,
    uint pickID)
{
    if(pickID == 1u)
        return SampleForward(x1, n1, matID, o, waveSeed, threadSeed);
    return SampleBackwardNEE(x1, n1, matID, o, waveSeed, threadSeed); // backward sampler is fallback
}

inline float BD_OneSamplePDF(float pdf_c, uint pickID, SampleData sdata, BDReturn bdreturn, float2 sprobs)
{
    if(pdf_c == 0.0f) return 0.0f;
    // Compute only the missing pdf
    float pdf_other = 0.0f;

    if (pickID == 1u) {
        // Current = Forward, Other = BSDF+NEE
        pdf_other = max(0.0f, PdfFromBackwardNEE(sdata.x1, sdata.n1, sdata.matID, sdata.o, bdreturn));
        // mixture: q_fwd * p_fwd (pdf_c) + q_bsdf * p_bsdf
        return max(0.0f, sprobs.y * pdf_c + sprobs.x * pdf_other);
    } else { // pickID == 0u
        // Current = BSDF+NEE, Other = Forward
        pdf_other = max(0.0f, PdfFromForward(sdata.x1, bdreturn));
        // mixture: q_bsdf * p_bsdf (pdf_c) + q_fwd * p_fwd
        return max(0.0f, sprobs.x * pdf_c + sprobs.y * pdf_other);
    }
}