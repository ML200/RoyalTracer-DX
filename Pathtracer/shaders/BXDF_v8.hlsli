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
inline uint SelectSamplingStrategy(PathState pstate, inout RandomData rdata)
{
    SamplingP p = CalculateStrategyProbabilities(pstate.matID, pstate.o, pstate.n_s);
    // Draw
    float r = RandomFloatSingle(rdata.seed.x);

    // CDF
    float c = p.Pspec;                 if (r < c) return 1;  // spec
    return 0; // diffuse
}


// Sample the BRDF of the given strategy
inline float3 SampleBRDF(PathState pstate, inout RandomData rdata) {
    // Select one method
    uint strategy = SelectSamplingStrategy(pstate, rdata);
    float3 sample;

    // TODO eta management ior of incoming and transmitted
    float etat = materials[pstate.matID].Ni;
    float etai = pstate.ior_pointer >= 0 ? pstate.ior_stack[pstate.ior_pointer] : 1.0f;

    bool refract = false;

    //Sample from the selected strategy
    if(strategy == 0){ // diffuse
        sample = SampleBRDF_Lambertian(pstate.matID, pstate.o, pstate.n_s, pstate.n_g, rdata.seed);
    }
    else if(strategy == 1){ // specular
        sample = SampleBRDF_GGX(pstate.matID, pstate.o, pstate.n_s, pstate.n_g, etai, etat, refract, rdata.seed);
    }
    else{
        sample = SampleBRDF_Lambertian(pstate.matID, pstate.o, pstate.n_s, pstate.n_g, rdata.seed);
    }

    // Reject invalid (below surface) samples
    if(refract){
        if(dot(sample, pstate.n_g) > 0.0f) return (float3)0.0f;
    }
    else{
        if(dot(sample, pstate.n_g) <= 0.0f) return (float3)0.0f;
    }

    return sample;
}


// Evaluation and pdf for the complete material model
inline float3 EvaluateBRDF_COMBINED(PathState pstate, SampleState sstate)
{
    float gate = 1.0f;
    float T = 1.0f;

    float3 f = 0.0.xxx;

    // ior of incoming and transmitted
    float etat = materials[pstate.matID].Ni;
    float etai = pstate.ior_pointer >= 0 ? pstate.ior_stack[pstate.ior_pointer] : 1.0f;

    // Base SPECULAR
    float3 f_spec = EvaluateBRDF_GGX(pstate.matID, pstate.n_s, pstate.n_g, -sstate.s, pstate.o, etai, etat);    // spec
    f += gate * f_spec;
    gate *= Transmittance_GGX(pstate.matID, pstate.n_s, -sstate.s, pstate.o, etai, etat);
    // Base DIFFUSE
    float3 f_diff = EvaluateBRDF_Lambertian(pstate.matID, pstate.n_s, pstate.n_g, -sstate.s, pstate.o, etai, etat); // diff
    f += gate * f_diff;

    return f;
}

inline float BRDF_PDF_COMBINED(PathState pstate, SampleState sstate)
{
    SamplingP p = CalculateStrategyProbabilities(pstate.matID, pstate.o, pstate.n_s);

    // TODO eta management ior of incoming and transmitted
    float etat = materials[pstate.matID].Ni;
    float etai = pstate.ior_pointer >= 0 ? pstate.ior_stack[pstate.ior_pointer] : 1.0f;

    float pd = BRDF_PDF_Lambertian(pstate.matID, pstate.n_s, pstate.n_g, -sstate.s, pstate.o);
    float ps = BRDF_PDF_GGX(pstate.matID, pstate.n_s, pstate.n_g, -sstate.s, pstate.o, etai, etat);

    return p.Pdiff * pd + p.Pspec * ps;
}