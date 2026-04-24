//====================================
//LAYERED BXDF
//====================================

struct SamplingP{
    float Psheen;
    float Pcoat;
    float Pspec;
    float Pdiff;
};

//====================================
//STRATEGY PROBABILITIES
//====================================
//energy cascade approximating actual distribution, cheap
inline SamplingP CalculateStrategyProbabilities(uint mID, float3 outgoing, float3 normal, float etai, float etat, float3 Kd, float Pm)
{
    float r_sheen = Sampling_Weight_SHEEN(mID, normal, outgoing);
    float r_coat  = Sampling_Weight_COAT(mID, normal, outgoing, etai, etat);
    float r_ggx   = Sampling_Weight_GGX(mID, normal, outgoing, etai, etat, Kd, Pm);
    float r_lamb   = Sampling_Weight_Lambertian(mID, normal, outgoing);

    SamplingP sp;

    sp.Psheen = r_sheen;
    float energy_after_sheen = 1.0f - sp.Psheen;
    sp.Pcoat = energy_after_sheen * r_coat * 3.0f;
    float energy_after_coat = energy_after_sheen * (1.0f - r_coat);
    sp.Pspec = energy_after_coat * r_ggx * 5.0f; //boost specular for better denoising
    float energy_after_ggx = energy_after_coat * (1.0f - r_ggx);
    sp.Pdiff = energy_after_ggx * r_lamb;

    float total_prob = sp.Psheen + sp.Pcoat + sp.Pspec + sp.Pdiff;
    if (total_prob > 0.0f)
    {
        sp.Psheen /= total_prob;
        sp.Pcoat  /= total_prob;
        sp.Pspec  /= total_prob;
        sp.Pdiff  /= total_prob;
    }
    else
    {
        sp.Psheen = 0.0f;
        sp.Pcoat  = 0.0f;
        sp.Pspec  = 0.0f;
        sp.Pdiff  = 1.0f;
    }
    return sp;
}


//====================================
//STRATEGY SELECTION
//====================================
//0=Diffuse, 1=GGX, 2=Coat, 3=Sheen
inline uint SelectSamplingStrategy(SamplingP p, inout uint seed)
{
    float r = RandomFloatSingle(seed);

    float c = p.Pdiff;
    if (r < c) return 0;
    c += p.Pspec;
    if (r < c) return 1;
    c += p.Pcoat;
    if (r < c) return 2;
    return 3;
}


//====================================
//BRDF SAMPLING
//====================================
inline float3 SampleBRDF(SamplingP p, uint matID, float3 o, float3 n_s, float3 n_g, float3 localKd, float localPr, float localPm, inout uint seed, float etai, float etat) {
    uint strategy = SelectSamplingStrategy(p, seed);
    float3 sample;

    bool refract = false;
    const bool canRefract = true;

    if(strategy == 0){
        sample = SampleBRDF_Lambertian(matID, o, n_s, n_g, seed);
    }
    else if(strategy == 1){
        sample = SampleBRDF_GGX(matID, o, n_s, n_g, etai, etat, refract, seed, localKd, localPr, localPm, canRefract);
    }
    else if(strategy == 2){
        sample = SampleBRDF_COAT(matID, o, n_s, n_g, seed);
    }
    else if(strategy == 3){
        sample = SampleBRDF_SHEEN(matID, o, n_s, n_g, seed);
    }
    else{
        sample = SampleBRDF_Lambertian(matID, o, n_s, n_g, seed);
    }

    //reject below-surface samples
    float  Ng_wi  = dot(sample, n_g);

    if (!refract) {
        if (Ng_wi <= 0.0f) {
            sample = reflect(sample, n_g);
            Ng_wi  = -Ng_wi;
        }
    } else {
        if (Ng_wi >= 0.0f) {
            sample = reflect(sample, n_g);
            Ng_wi  = -Ng_wi;
        }
    }

    return sample;
}


//====================================
//COMBINED BRDF DATA
//====================================
struct BrdfData {
    float3 val;
    float pdf;
};

//pdf-only for callers that don't need BRDF val
//matches EvaluateAndPdf_COMBINED's pdf math, zero-weight lobes skip helper
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

//====================================
//COMBINED BRDF EVALUATION
//====================================
//each lobe branches on sampling prob, skips eval/pdf/transmittance when p.X < EPSILON
//SER keeps branches coherent across similar materials
//individual lobes return transmittance=1 when weight ~0
inline float3 EvaluateBRDF_COMBINED(
    SamplingP p,
    uint matID, float3 n_s, float3 n_g, float3 s, float3 o,
    float3 localKd, float localPr, float localPm, float etai, float etat)
{
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
        const CoatResult cr = EvalCoatAll(matID, N, V, L, etai, etat);
        f    += gate * cr.f;
        gate *= cr.t;
    }
    if (p.Pspec >= EPSILON) {
        const GGXResult gr = EvalGGXAll(matID, N, fN, V, L, etai, etat, localKd, localPr, localPm);
        f    += gate * gr.f;
        gate *= gr.t;
    }
    if (p.Pdiff >= EPSILON) {
        f += gate * EvaluateBRDF_Lambertian(matID, n_s, n_g, -s, o, etai, etat, localKd);
    }
    return f;
}

//====================================
//FUSED EVAL AND PDF
//====================================
inline BrdfData EvaluateAndPdf_COMBINED(
    SamplingP p,
    uint matID, float3 n_s, float3 n_g, float3 s, float3 o,
    float3 localKd, float localPr, float localPm, float etai, float etat)
{
    BrdfData res;
    res.val = 0.0f;
    res.pdf = 0.0f;

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
        const CoatResult cr = EvalCoatAll(matID, N, V, L, etai, etat);
        res.val += gate * cr.f;
        res.pdf += p.Pcoat * cr.pdf;
        gate    *= cr.t;
    }
    if (p.Pspec >= EPSILON) {
        const GGXResult gr = EvalGGXAll(matID, N, fN, V, L, etai, etat, localKd, localPr, localPm);
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
