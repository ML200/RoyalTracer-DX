/*
Manager for the bxdf evaluation
*/

struct SamplingP{
    float Psheen;
    float Pcoat;
    float Pspec;
    float Pdiff;
    float Ptrans;
};

inline SamplingP CalculateStrategyProbabilities(uint mID, float3 outgoing, float3 normal)
{
    // Sum of all weights
    float w_sum = 0.0f;

    // TODO

    SamplingP sp;
    sp.Pspec = 0.0f;
    sp.Pdiff = 1.0f;
    return sp;
}


// Select a sampling strategy for the given material:
// 0 - Diffuse
// 1 - Specular GGX
// 2 - Clearcoat
// 3 - Sheen
// 4 - Transmission
inline uint SelectSamplingStrategy(uint mID, float3 outgoing, float3 normal, inout RandomData rdata)
{
    SamplingP p = CalculateStrategyProbabilities(mID, outgoing, normal);
    // Draw
    float r = RandomFloatSingle(rdata.seed.x);

    // CDF
    float c = p.Pspec;                 if (r < c) return 1;  // spec
    return 0; // diffuse
}


// Sample the BRDF of the given strategy
inline float3 SampleBRDF(uint mID, float3 outgoing, float3 flatNormal, float3 normal, inout RandomData rdata) {
    // Select one method
    uint strategy = SelectSamplingStrategy(mID, outgoing, normal, rdata);

    //Sample from the selected strategy
    if(strategy == 0){ // diffuse
        return SampleBRDF_Lambertian(mID, outgoing, normal, flatNormal, rdata.seed);
    }
    else if(strategy == 1){ // specular
        return SampleBRDF_GGX(mID, outgoing, normal, flatNormal, rdata.seed);
    }

    // fallback
    return SampleBRDF_Lambertian(mID, outgoing, normal, flatNormal, rdata.seed);
}


// Evaluation and pdf for the complete material model
inline float3 EvaluateBRDF_COMBINED(uint mID, float3 N, float3 wi, float3 wo)
{
    float gate = 1.0f;
    float T = 1.0f;

    float3 f = 0.0.xxx;

    // Base SPECULAR
    /*float3 f_spec = EvaluateBRDF_GGX_Dielectric(mID, N, wi, wo, 1.0f, 1.0f);    // spec
    f += gate * f_spec;
    gate *= Transmittance_GGX_Dielectric(mID, N, wi, wo, 1.0f, 1.0f);*/

    // Base DIFFUSE
    float3 f_diff = EvaluateBRDF_Lambertian(mID, N, wi, wo, 1.0f, 1.0f); // diff
    f += gate * f_diff;
    gate *= Transmittance_Lambertian(mID, N, wi, wo, 1.0f, 1.0f);

    return f;
}

inline float BRDF_PDF_COMBINED(uint mID, float3 N, float3 wi, float3 wo)
{
    SamplingP p = CalculateStrategyProbabilities(mID, wo, N);

    float pd = BRDF_PDF_Lambertian(mID, N, wi, wo);
    float ps = BRDF_PDF_GGX(mID, N, wi, wo);

    return p.Pdiff * pd + p.Pspec;
}
