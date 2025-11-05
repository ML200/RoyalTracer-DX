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
    sp.Pspec = 0.5f;
    sp.Pdiff = 0.5f;
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
    float3 sample;

    // TODO eta management ior of incoming and transmitted
    float etat = materials[mID].Ni;
    float etai = 1.0f;

    //Sample from the selected strategy
    if(strategy == 0){ // diffuse
        sample = SampleBRDF_Lambertian(mID, outgoing, normal, flatNormal, rdata.seed);
    }
    else if(strategy == 1){ // specular
        sample = SampleBRDF_GGX(mID, outgoing, normal, flatNormal, etai, etat, rdata.seed);
    }
    else{
        sample = SampleBRDF_Lambertian(mID, outgoing, normal, flatNormal, rdata.seed);
    }

    // Check that the sampled direction is indeed on the right side of the surface
    /*if(dot(sample, flatNormal) <= 0.0f)
        return (float3)0.0f;*/
    return sample;
}


// Evaluation and pdf for the complete material model
inline float3 EvaluateBRDF_COMBINED(uint mID, float3 Ng, float3 N, float3 wi, float3 wo)
{
    float gate = 1.0f;
    float T = 1.0f;

    float3 f = 0.0.xxx;

    // ior of incoming and transmitted
    float etat = materials[mID].Ni;
    float etai = 1.0f;

    // Base SPECULAR
    float3 f_spec = EvaluateBRDF_GGX(mID, N, Ng, wi, wo, etai, etat);    // spec
    f += gate * f_spec;
    gate *= Transmittance_GGX(mID, N, wi, wo, etai, etat);
    // Base DIFFUSE
    float3 f_diff = EvaluateBRDF_Lambertian(mID, N, Ng, wi, wo, etai, etat); // diff
    f += gate * f_diff;

    return f;
}

inline float BRDF_PDF_COMBINED(uint mID, float3 Ng, float3 N, float3 wi, float3 wo)
{
    SamplingP p = CalculateStrategyProbabilities(mID, wo, N);

    // TODO eta management ior of incoming and transmitted
    float etat = materials[mID].Ni;
    float etai = 1.0f;

    float pd = BRDF_PDF_Lambertian(mID, N, Ng, wi, wo);
    float ps = BRDF_PDF_GGX(mID, N, Ng, wi, wo, etai, etat);

    return p.Pdiff * pd + p.Pspec * ps;
}