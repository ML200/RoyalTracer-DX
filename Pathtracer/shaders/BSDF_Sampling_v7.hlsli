// Sample a BSDF sample
SampleReturn SampleBSDF(
    SampleData sdata,
    inout uint waveSeed,
    inout uint2 threadSeed
){
    // Sample a BSDF direction
    float3 sample;
    uint strategy = SelectSamplingStrategy(sdata.matID, sdata.o, sdata.n1, threadSeed);
    SampleBRDF(strategy, sdata.matID, sdata.o, sdata.n1, sdata.n1, sample, sdata.x1, threadSeed);

    if(all(sample == 0.0f))
        return (SampleReturn)0;

    // Trace the ray
    RayDesc ray;
    ray.Origin = sdata.x1;
    ray.Direction = sample;
    ray.TMin = 0.001f;
    ray.TMax = 10000;
    HitInfo samplePayload;
    TraceRayInline_HitInfo(SceneBVH, ray, samplePayload, RAY_FLAG_NONE, 0xFF);

    // Evaluate the contribution
    float3 emission = materials[samplePayload.materialID].Ke;
    float pdf_b = 0.f;
    float pdf_l = 0.f;

    //if(any(emission > 0.0f)){
        float2 probs = CalculateStrategyProbabilities(sdata.matID, sdata.o, sdata.n1);
        float pdf0 = BRDF_PDF(0, sdata.matID, sdata.n1, -sample, sdata.o);
        float pdf1 = BRDF_PDF(1, sdata.matID, sdata.n1, -sample, sdata.o);
        float P1 = probs.x * pdf0;
        float P2 = probs.y * pdf1;

        float3 L =  sdata.x1 - samplePayload.hitPosition;
        float cos_light = dot(samplePayload.hitNormal, normalize(L));
        if(cos_light<0.0f)
            return (SampleReturn)0;
        float dist = length(L);
        float dist2 = dist * dist;

        pdf_b = (P1 + P2) * cos_light / dist2;
        pdf_l = LT_Pdf_LightTree_Area(sdata.x1, sdata.n1, samplePayload.lightID, samplePayload.objID);//((emission.x + emission.y + emission.z) / 3.0f) / g_EmissiveTriangles[0].total_weight;
    //}

    // Fill in the sample and return
    SampleReturn sreturn = (SampleReturn)0;
    sreturn.x2 = samplePayload.hitPosition;
    sreturn.n2 = samplePayload.hitNormal;
    sreturn.L2 = emission;
    sreturn.objID = samplePayload.objID;
    sreturn.matID = samplePayload.materialID;

    sreturn.pdf_bsdf = pdf_b;
    sreturn.pdf_nee = pdf_l;

    return sreturn;
}

#ifdef ENABLE_RAY_QUERY_INLINE
// Sample a BSDF sample (generalized for solid angle)
SampleReturn SampleBSDF_gen(
    in float3 x1,
    in float3 n1,
    in uint matID,
    in float3 o,
    inout uint waveSeed,
    inout uint2 threadSeed
){
    // Sample a BSDF direction
    float3 sample;
    uint strategy = SelectSamplingStrategy(matID, o, n1, threadSeed);
    SampleBRDF(strategy, matID, o, n1, n1, sample, x1, threadSeed);

    if(all(sample == 0.0f))
        return (SampleReturn)0;

    // Trace the ray
    RayDesc ray;
    ray.Origin = x1;
    ray.Direction = sample;
    ray.TMin = 0.01f;
    ray.TMax = 10000.0f;
    HitInfo samplePayload;
    TraceRayInline_HitInfo(SceneBVH, ray, samplePayload, RAY_FLAG_NONE, 0xFF);

    // Evaluate the contribution
    float3 emission = materials[samplePayload.materialID].Ke;
    float pdf_b = 0.f;
    float pdf_l = 0.f;

    //if(any(emission > 0.0f)){
        float2 probs = CalculateStrategyProbabilities(matID, o, n1);
        float pdf0 = BRDF_PDF(0, matID, n1, -sample, o);
        float pdf1 = BRDF_PDF(1, matID, n1, -sample, o);
        float P1 = SafeMultiply(probs.x, pdf0);
        float P2 = SafeMultiply(probs.y, pdf1);

        float3 L =  x1 - samplePayload.hitPosition;
        float cos_light = dot(samplePayload.hitNormal, normalize(L));
        if(cos_light<0.0f){
            if(any(emission > 0.0f))
                return (SampleReturn)0;
            else{
                samplePayload.hitNormal = -samplePayload.hitNormal;
                cos_light = dot(samplePayload.hitNormal, normalize(L));
            }
        }

        float dist = length(L);
        float dist2 = dist * dist;

        pdf_b = (P1 + P2);
        pdf_l = LT_Pdf_LightTree_Area(x1, n1, samplePayload.lightID, samplePayload.objID) * dist2 / cos_light;//((emission.x + emission.y + emission.z) / 3.0f) / g_EmissiveTriangles[0].total_weight * dist2 / cos_light;
    //}

    // Fill in the sample and return
    SampleReturn sreturn = (SampleReturn)0;
    sreturn.x2 = samplePayload.hitPosition;
    sreturn.n2 = samplePayload.hitNormal;
    sreturn.L2 = emission;
    sreturn.objID = samplePayload.objID;
    sreturn.matID = samplePayload.materialID;

    sreturn.pdf_bsdf = pdf_b;
    sreturn.pdf_nee = pdf_l;

    return sreturn;
}
#endif