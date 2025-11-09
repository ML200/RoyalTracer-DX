/*
Manager for the bxdf evaluation
*/

struct SamplingP{
    float Psheen;
    float Pcoat;
    float Pspec;
    float Pdiff;
};

inline SamplingP CalculateStrategyProbabilities(uint mID, float3 outgoing, float3 normal)
{
    // Sum of all weights
    float w_sum = 0.0f;

    // TODO

    SamplingP sp;
    sp.Pspec = 0.5f;
    sp.Pdiff = 0.3f;
    sp.Psheen = 0.1f;
    sp.Pcoat = 0.1f;
    return sp;
}


// Select a sampling strategy for the given material:
// 0 - Diffuse
// 1 - Specular GGX
// 2 - Clearcoat
// 3 - Sheen
inline uint SelectSamplingStrategy(PathState pstate, inout RandomData rdata)
{
    SamplingP p = CalculateStrategyProbabilities(pstate.matID, pstate.o, pstate.n_s);
    float r = RandomFloatSingle(rdata.seed.x);
    float c = p.Pdiff;
    if (r < c) return 0;                 // Diffuse
    c += p.Pspec;
    if (r < c) return 1;                 // GGX
    c += p.Pcoat;
    if (r < c) return 2;                 // Coat
    return 3;                            // Sheen
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
    else if(strategy == 2){ // coat
        sample = SampleBRDF_COAT(pstate.matID, pstate.o, pstate.n_s, pstate.n_g, rdata.seed);
    }
    else if(strategy == 3){ // sheen
        sample = SampleBRDF_SHEEN(pstate.matID, pstate.o, pstate.n_s, pstate.n_g, rdata.seed);
    }
    else{
        sample = SampleBRDF_Lambertian(pstate.matID, pstate.o, pstate.n_s, pstate.n_g, rdata.seed);
    }

    // Reject invalid (below surface) samples
    float  Ng_wi  = dot(sample, pstate.n_g);

    if (!refract) {
        // reflection
        if (Ng_wi <= 0.0f) {
            sample = reflect(sample, pstate.n_g);
            Ng_wi  = -Ng_wi;
        }
    } else {
        // transmission wants
        if (Ng_wi >= 0.0f) {
            sample = reflect(sample, pstate.n_g);
            Ng_wi  = -Ng_wi;
        }
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

    // Base SHEEN
    float3 f_sheen = EvaluateBRDF_SHEEN(pstate.matID, pstate.n_s, -sstate.s, pstate.o);
    f += gate * f_sheen;
    gate *= Transmittance_SHEEN(pstate.matID, pstate.n_s, -sstate.s, pstate.o);
    // Base COAT
    float3 f_coat = EvaluateBRDF_COAT(pstate.matID, pstate.n_s, -sstate.s, pstate.o, etai, 1.5f);    // coat
    f += gate * f_coat;
    gate *= Transmittance_COAT(pstate.matID, pstate.n_s, -sstate.s, pstate.o, etai, etat);
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
    float pc = BRDF_PDF_COAT(pstate.matID, pstate.n_s, -sstate.s, pstate.o, etai, etat);
    float psh = BRDF_PDF_SHEEN(pstate.matID, pstate.n_s, -sstate.s, pstate.o);

    return p.Pdiff * pd + p.Pspec * ps + p.Psheen * psh + p.Pcoat * pc;
}