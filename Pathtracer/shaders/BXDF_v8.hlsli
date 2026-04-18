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
    sp.Pcoat = energy_after_sheen * r_coat * 3.0f;
    float energy_after_coat = energy_after_sheen * (1.0f - r_coat);
    sp.Pspec = energy_after_coat * r_ggx * 5.0f; // cheap boost for specular surfaces -> better denosining
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
inline float3 SampleBRDF(SamplingP p, uint matID, float3 o, float3 n_s, float3 n_g, float3 localKd, float localPr, float localPm, inout uint seed, float etai, float etat) {
    // Select one method
    uint strategy = SelectSamplingStrategy(p, seed);
    float3 sample;

    bool refract = false;
    const bool canRefract = true;

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


// Small helper struct for combined data
struct BrdfData {
    float3 val;
    float pdf;
};

// KEEP — pdf-only helper for callers that don't need the BRDF value.
// Matches EvaluateAndPdf_COMBINED's pdf math: zero-weight lobes contribute
// nothing to the sum, so the corresponding BRDF_PDF_* helper is skipped.
inline float BRDF_PDF_COMBINED(
    SamplingP p,
    uint matID, float3 n_s, float3 n_g, float3 s, float3 o,
    float3 localKd, float localPr, float localPm, float etai, float etat)
{
    float pdf = 0.0f;
    if (p.Psheen >= EPSILON)
        pdf += p.Psheen * BRDF_PDF_SHEEN(matID, n_s, -s, o);
    if (p.Pcoat >= EPSILON)
        pdf += p.Pcoat  * BRDF_PDF_COAT(matID, n_s, -s, o, etai, etat);
    if (p.Pspec >= EPSILON)
        pdf += p.Pspec  * BRDF_PDF_GGX(matID, n_s, n_g, -s, o, etai, etat, localKd, localPr, localPm);
    if (p.Pdiff >= EPSILON)
        pdf += p.Pdiff  * BRDF_PDF_Lambertian(matID, n_s, n_g, -s, o);
    return pdf;
}

// Evaluation for the complete material model.
// Each lobe branches on its own sampling probability — when p.X < EPSILON
// the lobe's evaluation, pdf, and transmittance update are all skipped.
// SER keeps the branches coherent across materials with similar profiles.
// Gate-propagation is safe because the individual material helpers return
// transmittance = 1 when their raw weight is ~0 (Sheen/Coat early-outs in
// their own impls; GGX corner-cases fall within numerical tolerance).
inline float3 EvaluateBRDF_COMBINED(
    SamplingP p,
    uint matID, float3 n_s, float3 n_g, float3 s, float3 o,
    float3 localKd, float localPr, float localPm, float etai, float etat)
{
    Material mat = materials[matID];
    const float3 N  = normalize(n_s);
    const float3 fN = normalize(n_g);
    const float3 V  = normalize(o);
    const float3 L  = normalize(s);

    float  gate = 1.0f;
    float3 f    = 0.0f;

    if (p.Psheen >= EPSILON) {
        f    += gate * EvaluateBRDF_SHEEN(matID, n_s, -s, o);
        gate *= Transmittance_SHEEN(matID, n_s, -s, o);
    }
    if (p.Pcoat >= EPSILON) {
        const CoatResult cr = EvalCoatAll(mat, N, V, L, etai, etat);
        f    += gate * cr.f;
        gate *= cr.t;
    }
    if (p.Pspec >= EPSILON) {
        const GGXResult gr = EvalGGXAll(mat, N, fN, V, L, etai, etat, localKd, localPr, localPm);
        f    += gate * gr.f;
        gate *= gr.t;
    }
    if (p.Pdiff >= EPSILON) {
        f += gate * EvaluateBRDF_Lambertian(matID, n_s, n_g, -s, o, etai, etat, localKd);
    }
    return f;
}

// Fused eval + pdf — same branching discipline as EvaluateBRDF_COMBINED.
// The pdf sum is p.Pdiff*pd + p.Pspec*ps + p.Pcoat*pc + p.Psheen*psh, so a
// zero weight nulls that lobe's pdf contribution and its pdf helper can be
// skipped outright.
inline BrdfData EvaluateAndPdf_COMBINED(
    SamplingP p,
    uint matID, float3 n_s, float3 n_g, float3 s, float3 o,
    float3 localKd, float localPr, float localPm, float etai, float etat)
{
    BrdfData res;
    res.val = 0.0f;
    res.pdf = 0.0f;

    Material mat = materials[matID];
    const float3 N  = normalize(n_s);
    const float3 fN = normalize(n_g);
    const float3 V  = normalize(o);
    const float3 L  = normalize(s);

    float gate = 1.0f;

    if (p.Psheen >= EPSILON) {
        res.val += gate * EvaluateBRDF_SHEEN(matID, n_s, -s, o);
        res.pdf += p.Psheen * BRDF_PDF_SHEEN(matID, n_s, -s, o);
        gate    *= Transmittance_SHEEN(matID, n_s, -s, o);
    }
    if (p.Pcoat >= EPSILON) {
        const CoatResult cr = EvalCoatAll(mat, N, V, L, etai, etat);
        res.val += gate * cr.f;
        res.pdf += p.Pcoat * cr.pdf;
        gate    *= cr.t;
    }
    if (p.Pspec >= EPSILON) {
        const GGXResult gr = EvalGGXAll(mat, N, fN, V, L, etai, etat, localKd, localPr, localPm);
        res.val += gate * gr.f;
        res.pdf += p.Pspec * gr.pdf;
        gate    *= gr.t;
    }
    if (p.Pdiff >= EPSILON) {
        res.val += gate * EvaluateBRDF_Lambertian(matID, n_s, n_g, -s, o, etai, etat, localKd);
        res.pdf += p.Pdiff * BRDF_PDF_Lambertian(matID, n_s, n_g, -s, o);
    }
    return res;
}