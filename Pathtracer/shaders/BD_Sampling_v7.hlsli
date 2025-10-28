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

    // Check that the reconnetion is even possible (not behind surface, not on the same surface etc)
    if(dot(n1, normalize(samplePayload.hitPosition - x1)) < EPSILON) return (BDReturn)0.0f;

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
    BDReturn sreturn = (BDReturn)0; // return data. We fill the possible reconnections early and return on failure.

    //Sample a bsdf ray and eval its pdf
    float3 sampleBSDF;
    uint strategy = SelectSamplingStrategy(matID, o, n1, threadSeed);
    SampleBRDF(strategy, matID, o, n1, n1, sampleBSDF, x1, threadSeed);
    if(all(sampleBSDF == 0.0f)) return sreturn;

    RayDesc ray;
    ray.Origin = x1;
    ray.Direction = sampleBSDF;
    ray.TMin = 0.001f;
    ray.TMax = 10000.0f;
    HitInfo samplePayload;
    TraceRayInline_HitInfo(SceneBVH, ray, samplePayload, RAY_FLAG_NONE, 0xFF);
    if(any(materials[samplePayload.materialID].Ke > 0.0f) || all(samplePayload.hitNormal< EPSILON)) return sreturn;
    float pdfBSDF = BRDF_PDF_COMBINED(matID, n1, -sampleBSDF, o);

    // Some intermediate variables:
    float3 x2 = samplePayload.hitPosition;
    float3 n2 = samplePayload.hitNormal;
    uint matID2 = samplePayload.materialID;
    float3 o2 = x1 - x2;

    // relevant data to reuse this segment to advance the path
    sreturn.x2 = x2;
    sreturn.n2 = n2;
    sreturn.matID = matID2;
    sreturn.pdf_seg = pdfBSDF;


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
    if(V == 0.0f) return sreturn;

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
        return sreturn;
    }

    // Compute NEE PDF (solid angle!)
    float pdfNEE = pick.pdf / max(area_L, 1e-10f) * dist2 / max(dot(n3, -L_norm), 0.0f);


    //return the sample. We evaluate it using the extended reconnection
    sreturn.L2 = g_EmissiveTriangles[lightIdx].emission;
    sreturn.objID = samplePayload.objID;
    sreturn.x3 = x3;
    sreturn.n3 = n3;
    sreturn.triID = lightIdx;

    sreturn.pdf = pdfBSDF * pdfNEE;

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


#ifdef ENABLE_RAY_QUERY_INLINE
// Sample a BSDF + BSDF sample
BDReturn SampleBackwardBSDF(
    in float3 x1,
    in float3 n1,
    in uint matID,
    in float3 o,
    inout uint waveSeed,
    inout uint2 threadSeed
){
    BDReturn sreturn = (BDReturn)0;

    //Sample a bsdf ray and eval its pdf
    float3 sampleBSDF;
    uint strategy = SelectSamplingStrategy(matID, o, n1, threadSeed);
    SampleBRDF(strategy, matID, o, n1, n1, sampleBSDF, x1, threadSeed);
    if(all(sampleBSDF == 0.0f)) return sreturn;

    RayDesc ray;
    ray.Origin = x1;
    ray.Direction = sampleBSDF;
    ray.TMin = 0.001f;
    ray.TMax = 10000.0f;
    HitInfo samplePayload;
    TraceRayInline_HitInfo(SceneBVH, ray, samplePayload, RAY_FLAG_NONE, 0xFF);
    if(length(materials[samplePayload.materialID].Ke) > 0.0f) return sreturn;
    float pdfBSDF1 = BRDF_PDF_COMBINED(matID, n1, -sampleBSDF, o);

    // Some intermediate variables:
    float3 x2 = samplePayload.hitPosition;
    float3 n2 = samplePayload.hitNormal;
    uint matID2 = samplePayload.materialID;
    float3 o2 = normalize(x1 - x2);

    sreturn.x2 = x2;
    sreturn.n2 = n2;
    sreturn.matID = matID2;
    sreturn.pdf_seg = pdfBSDF1;

    // Sample an second BSDF ray and eval its pdf
    float3 sampleBSDF2;
    uint strategy2 = SelectSamplingStrategy(matID2, o2, n2, threadSeed);
    SampleBRDF(strategy2, matID2, o2, n2, n2, sampleBSDF2, x2, threadSeed);
    if(all(sampleBSDF2 == 0.0f)) return sreturn; // Invalid sample

    RayDesc ray2;
    ray2.Origin = x2;
    ray2.Direction = sampleBSDF2;
    ray2.TMin = 0.001f;
    ray2.TMax = 10000.0f;
    HitInfo samplePayload2;
    TraceRayInline_HitInfo(SceneBVH, ray2, samplePayload2, RAY_FLAG_NONE, 0xFF);
    if(length(materials[samplePayload2.materialID].Ke) < EPSILON) return sreturn; // no light hit -> bad sample
    if(dot(samplePayload2.hitNormal, sampleBSDF2) >= 0.0f) return sreturn;
    float pdfBSDF2 = BRDF_PDF_COMBINED(matID2, n2, -sampleBSDF2, o2);

    //return the sample. We evaluate it using the extended reconnection
    sreturn.L2 = materials[samplePayload2.materialID].Ke;
    sreturn.objID = samplePayload.objID;
    sreturn.x3 = samplePayload2.hitPosition;
    sreturn.n3 = samplePayload2.hitNormal;
    sreturn.triID = samplePayload2.lightID;

    sreturn.pdf = pdfBSDF1 * pdfBSDF2;

    return sreturn;
}
#endif


inline float PdfFromBackwardBSDF(
    float3 x1,
    float3 n1,
    uint   matID1,
    float3 o1,
    BDReturn bd
){
    //segment x1 -> x2
    float3 w12   = normalize(bd.x2 - x1);

    float pdf1 = BRDF_PDF_COMBINED(matID1, n1, -w12, o1);
    if (pdf1 <= 0.0) return 0.0;

    //segment x2 -> x3
    float3 w23   = normalize(bd.x3 - bd.x2);
    float3 o2    = normalize(x1 - bd.x2);

    float pdf2 = BRDF_PDF_COMBINED(bd.matID, bd.n2, -w23, o2);
    if (pdf2 <= 0.0) return 0.0;

    return pdf1 * pdf2;
}



// HELPERS
// ----------------------------------------------------------------------
// Test function to sample one bsdf path segment to gain an unbiased path advancement method
#ifdef ENABLE_RAY_QUERY_INLINE
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

//-----------------------------------------------------------------------------------------------
inline float3 BD_MethodProbsFromRoughness(float roughness)
{
    /*float r = saturate(roughness);
    float t = saturate((r - 0.5f) * 2.0f);   // 0 at 0.5, 1 at 1.0

    // total mass for non-forward methods
    float p_bsdf_bucket = 1.0f - 0.5f * t;   // 1.0 -> 0.5
    float p_fwd         = 1.0f - p_bsdf_bucket;

    // 65/35 split inside the bsdf bucket
    const float nee_share = 0.65f;
    float p_bsdfnee  = p_bsdf_bucket * nee_share;
    float p_bsdfbsdf = p_bsdf_bucket * (1.0f - nee_share);

    // Sum should be 1, but keep a defensive normalize
    float sum = p_bsdfnee + p_fwd + p_bsdfbsdf;
    if (sum <= 0.0f) return float3(1.0f, 0.0f, 0.0f);

    return float3(p_bsdfnee, p_fwd, p_bsdfbsdf) / sum;*/
    return float3(0.4f, 0.4f, 0.2f);
}

inline uint BD_PickMethod(float3 probs, inout uint2 threadSeed)
{
    float xi = RandomFloatSingle(threadSeed.x);
    if (xi < probs.x) return 0u;               // BSDF+NEE
    xi -= probs.x;
    if (xi < probs.y) return 1u;               // Forward
    return 2u;                                  // BSDF+BSDF
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
    if (pickID == 1u)
        return SampleForward(x1, n1, matID, o, waveSeed, threadSeed);     // Forward
    else if (pickID == 2u)
        return SampleBackwardBSDF(x1, n1, matID, o, waveSeed, threadSeed); // BSDF+BSDF
    else
        return SampleBackwardNEE(x1, n1, matID, o, waveSeed, threadSeed);  // BSDF+NEE
}

inline float BD_OneSamplePDF(
    float pdf_c,               // pdf of the technique that actually generated the sample
    uint pickID,               // 0=BSDF+NEE, 1=Forward, 2=BSDF+BSDF
    SampleData sdata,
    BDReturn bdreturn,
    float3 sprobs)             // probs.x=BSDF+NEE, probs.y=Forward, probs.z=BSDF+BSDF
{
    if (pdf_c == 0.0f) return 0.0f;

    float p_bsdfnee  = 0.0f;
    float p_forward  = 0.0f;
    float p_bsdfbsdf = 0.0f;

    // Reuse the current technique pdf and compute the others
    if (pickID == 0u) {
        p_bsdfnee  = pdf_c;
        p_forward  = max(0.0f, PdfFromForward(sdata.x1, bdreturn));
        p_bsdfbsdf = max(0.0f, PdfFromBackwardBSDF(sdata.x1, sdata.n1, sdata.matID, sdata.o, bdreturn));
    } else if (pickID == 1u) {
        p_forward  = pdf_c;
        p_bsdfnee  = max(0.0f, PdfFromBackwardNEE(sdata.x1, sdata.n1, sdata.matID, sdata.o, bdreturn));
        p_bsdfbsdf = max(0.0f, PdfFromBackwardBSDF(sdata.x1, sdata.n1, sdata.matID, sdata.o, bdreturn));
    } else { // pickID == 2u
        p_bsdfbsdf = pdf_c;
        p_bsdfnee  = max(0.0f, PdfFromBackwardNEE(sdata.x1, sdata.n1, sdata.matID, sdata.o, bdreturn));
        p_forward  = max(0.0f, PdfFromForward(sdata.x1, bdreturn));
    }

    // Mixture pdf: sum over techniques
    float pdf_mix = sprobs.x * p_bsdfnee
                  + sprobs.y * p_forward
                  + sprobs.z * p_bsdfbsdf;

    return max(0.0f, pdf_mix);
}