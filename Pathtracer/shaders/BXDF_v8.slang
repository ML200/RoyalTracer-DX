// Layered BXDF evaluation and sampling

struct SamplingP{
    float Psheen;
    float Pcoat;
    float Pspec;
    float Pdiff;
};

inline SamplingP CalculateStrategyProbabilities(uint mID, float3 outgoing, float3 normal, float etai, float etat, float3 Kd, float Pm)
{
    float r_sheen = Sampling_Weight_SHEEN(mID, normal, outgoing);
    float r_coat  = Sampling_Weight_COAT(mID, normal, outgoing, etai, etat);
    float r_ggx   = Sampling_Weight_GGX(mID, normal, outgoing, etai, etat, Kd, Pm);
    float r_lamb   = Sampling_Weight_Lambertian(mID, normal, outgoing);

    // Energy cascade approximating actual energy distribution
    // While this isnt completely optimal, its cheap.
    SamplingP sp;

    sp.Psheen = r_sheen;
    float energy_after_sheen = 1.0f - sp.Psheen;
    sp.Pcoat = energy_after_sheen * r_coat;
    float energy_after_coat = energy_after_sheen * (1.0f - r_coat);
    sp.Pspec = energy_after_coat * r_ggx;
    float energy_after_ggx = energy_after_coat * (1.0f - r_ggx);
    sp.Pdiff = energy_after_ggx * r_lamb;

    // Safely normalize to ensure correct weights summing to 1!
    float total_prob = sp.Psheen + sp.Pcoat + sp.Pspec + sp.Pdiff;
    if (total_prob > 0.0f)
    {
        sp.Psheen /= total_prob;
        sp.Pcoat  /= total_prob;
        sp.Pspec  /= total_prob;
        sp.Pdiff  /= total_prob;
    }
    else // no div by 0
    {
        sp.Psheen = 0.0f;
        sp.Pcoat  = 0.0f;
        sp.Pspec  = 0.0f;
        sp.Pdiff  = 1.0f;
    }
    return sp;
}


// Select a sampling strategy for the given material:
// 0 - Diffuse
// 1 - Specular GGX
// 2 - Clearcoat
// 3 - Sheen
inline uint SelectSamplingStrategy(SamplingP p, inout uint seed)
{
    float r = RandomFloatSingle(seed);

    float c = p.Pdiff;
    if (r < c) return 0;                 // Diffuse
    c += p.Pspec;
    if (r < c) return 1;                 // GGX
    c += p.Pcoat;
    if (r < c) return 2;                 // Coat
    return 3;                            // Sheen
}


// Sample the BRDF of the given strategy
inline float3 SampleBRDF(SamplingP p, uint matID, float3 o, float3 n_s, float3 n_g, float3 localKd, float localPr, float localPm, inout uint seed, float etai, float etat, int ior_pointer) {
    // Select one method
    uint strategy = SelectSamplingStrategy(p, seed);
    float3 sample;

    bool refract = false;
    bool canRefract = ior_pointer < 3;

    //Sample from the selected strategy
    if(strategy == 0){ // diffuse
        sample = SampleBRDF_Lambertian(matID, o, n_s, n_g, seed);
    }
    else if(strategy == 1){ // specular
        sample = SampleBRDF_GGX(matID, o, n_s, n_g, etai, etat, refract, seed, localKd, localPr, localPm, canRefract);
    }
    else if(strategy == 2){ // coat
        sample = SampleBRDF_COAT(matID, o, n_s, n_g, seed);
    }
    else if(strategy == 3){ // sheen
        sample = SampleBRDF_SHEEN(matID, o, n_s, n_g, seed);
    }
    else{
        sample = SampleBRDF_Lambertian(matID, o, n_s, n_g, seed);
    }

    // Reject invalid (below surface) samples
    float  Ng_wi  = dot(sample, n_g);

    if (!refract) {
        // reflection
        if (Ng_wi <= 0.0f) {
            sample = reflect(sample, n_g);
            Ng_wi  = -Ng_wi;
        }
    } else {
        // transmission wants
        if (Ng_wi >= 0.0f) {
            sample = reflect(sample, n_g);
            Ng_wi  = -Ng_wi;
        }
    }

    return sample;
}


// Evaluation for the complete material model
inline float3 EvaluateBRDF_COMBINED(uint matID, float3 n_s, float3 n_g, float3 s, float3 o, float3 localKd, float localPr, float localPm, float etai, float etat)
{
    Material mat = materials[matID];
    float3 N  = normalize(n_s);
    float3 fN = normalize(n_g);
    float3 V  = normalize(o);
    float3 L  = normalize(s);

    float gate = 1.0f;
    float3 f = float3(0.0, 0.0, 0.0);

    // Base SHEEN
    float3 f_sheen = EvaluateBRDF_SHEEN(matID, n_s, -s, o);
    f += gate * f_sheen;
    gate *= Transmittance_SHEEN(matID, n_s, -s, o);

    // Base COAT (fused: eval+transmittance, pdf is DCE'd)
    {
        CoatResult cr = EvalCoatAll(mat, N, V, L, etai, etat);
        f += gate * cr.f;
        gate *= cr.t;
    }

    // Base SPECULAR (fused: eval+transmittance, pdf is DCE'd)
    {
        GGXResult gr = EvalGGXAll(mat, N, fN, V, L, etai, etat, localKd, localPr, localPm);
        f += gate * gr.f;
        gate *= gr.t;
    }

    // Base DIFFUSE
    float3 f_diff = EvaluateBRDF_Lambertian(matID, n_s, n_g, -s, o, etai, etat, localKd);
    f += gate * f_diff;

    return f;
}

// Combined PDF for all BXDF lobes
inline float BRDF_PDF_COMBINED(SamplingP p, uint matID, float3 n_s, float3 n_g, float3 s, float3 o, float3 localKd, float localPr, float localPm, float etai, float etat)
{
    float pd  = BRDF_PDF_Lambertian(matID, n_s, n_g, -s, o);
    float ps  = BRDF_PDF_GGX(matID, n_s, n_g, -s, o, etai, etat, localKd, localPr, localPm);
    float pc  = BRDF_PDF_COAT(matID, n_s, -s, o, etai, etat);
    float psh = BRDF_PDF_SHEEN(matID, n_s, -s, o);

    return p.Pdiff * pd + p.Pspec * ps + p.Psheen * psh + p.Pcoat * pc;
}

// Small helper struct for combined data
struct BrdfData {
    float3 val;
    float pdf;
};

inline BrdfData EvaluateAndPdf_COMBINED(
    SamplingP p,
    uint matID, float3 n_s, float3 n_g, float3 s, float3 o,
    float3 localKd, float localPr, float localPm, float etai, float etat)
{
    BrdfData res;
    res.val = 0.0f;
    res.pdf = 0.0f;

    Material mat = materials[matID];
    float3 N  = normalize(n_s);
    float3 fN = normalize(n_g);
    float3 V  = normalize(o);
    float3 L  = normalize(s);

    float gate = 1.0f;

    // Sheen
    {
        float3 f = EvaluateBRDF_SHEEN(matID, n_s, -s, o);
        float prob = BRDF_PDF_SHEEN(matID, n_s, -s, o);

        res.val += gate * f;
        res.pdf += p.Psheen * prob;

        gate *= Transmittance_SHEEN(matID, n_s, -s, o);
    }

    // Coat
    {
        CoatResult cr = EvalCoatAll(mat, N, V, L, etai, etat);
        res.val += gate * cr.f;
        res.pdf += p.Pcoat * cr.pdf;
        gate *= cr.t;
    }

    // GGX
    {
        GGXResult gr = EvalGGXAll(mat, N, fN, V, L, etai, etat, localKd, localPr, localPm);
        res.val += gate * gr.f;
        res.pdf += p.Pspec * gr.pdf;
        gate *= gr.t;
    }

    // Diffuse
    {
        float3 f = EvaluateBRDF_Lambertian(matID, n_s, n_g, -s, o, etai, etat, localKd);
        float prob = BRDF_PDF_Lambertian(matID, n_s, n_g, -s, o);

        res.val += gate * f;
        res.pdf += p.Pdiff * prob;
    }

    return res;
}