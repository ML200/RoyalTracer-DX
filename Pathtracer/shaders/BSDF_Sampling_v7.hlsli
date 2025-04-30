// Sample a NEE sample
SampleReturn SampleBSDF(
    SampleData sdata,
    inout uint waveSeed,
    inout uint2 threadSeed
){
    // Sample a BSDF direction
    float3 sample;
    uint strategy = SelectSamplingStrategy(sdata.matID, sdata.o, sdata.n1, threadSeed);
    SampleBRDF(strategy, sdata.matID, sdata.o, sdata.n1, sdata.n1, sample, sdata.x1, threadSeed);

    // Trace the ray
    RayDesc ray;
    ray.Origin = sdata.x1;
    ray.Direction = sample;
    ray.TMin = 0.001f;
    ray.TMax = 10000;
    HitInfo samplePayload;
    TraceRay(SceneBVH, RAY_FLAG_NONE, 0xFF, 0, 0, 0, ray, samplePayload);

    // Evaluate the contribution
    float3 emission = materials[samplePayload.materialID].Ke;
    float pdf_b = 0.f;
    float pdf_l = 0.f;

    if(any(emission > 0.0f)){
        float2 probs = CalculateStrategyProbabilities(sdata.matID, sdata.o, sdata.n1);
        float3 pdf0 = BRDF_PDF(0, sdata.matID, sdata.n1, -sample, sdata.o);
        float3 pdf1 = BRDF_PDF(1, sdata.matID, sdata.n1, -sample, sdata.o);
        float3 P1 = SafeMultiply(probs.x, pdf0);
        float3 P2 = SafeMultiply(probs.y, pdf1);

        float3 L =  sdata.x1 - samplePayload.hitPosition;
        float cos_light = dot(samplePayload.hitNormal, normalize(L));
        float dist = length(L);
        float dist2 = dist * dist;

        pdf_b = (P1 + P2) * cos_light / dist2;
        pdf_l = ((emission.x + emission.y + emission.z) / 3.0f) / g_EmissiveTriangles[0].total_weight;
    }

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