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
//cheap energy cascade approximating the actual distribution.
//etai/etat are IOR (~1..2.5), Pm is metalness (0..1). All fit cleanly in fp16,
//so the API takes half -- callers avoid promoting at the boundary. Lobe
//weight helpers still take float and get an implicit half->float at the call.
inline SamplingP CalculateStrategyProbabilities(uint mID, float3 outgoing, float3 normal, half etai, half etat, float3 Kd, half Pm)
{
    //all weights and energy fractions are in [0,1], boosted products max ~5, fp16 safe
    const half r_sheen = (half)Sampling_Weight_SHEEN(mID, normal, outgoing);
    const half r_coat  = (half)Sampling_Weight_COAT(mID, normal, outgoing, etai, etat);
    const half r_ggx   = (half)Sampling_Weight_GGX(mID, normal, outgoing, etai, etat, Kd, Pm);
    const half r_lamb  = (half)Sampling_Weight_Lambertian(mID, normal, outgoing);

    const half energy_after_sheen = (half)1.0 - r_sheen;
    const half energy_after_coat  = energy_after_sheen * ((half)1.0 - r_coat);
    const half energy_after_ggx   = energy_after_coat  * ((half)1.0 - r_ggx);

    SamplingP sp;
    sp.Psheen = (float)r_sheen;
    sp.Pcoat  = (float)(energy_after_sheen * r_coat * (half)3.0);
    sp.Pspec  = (float)(energy_after_coat  * r_ggx  * (half)5.0); //boosted to denoise better
    sp.Pdiff  = (float)(energy_after_ggx   * r_lamb);

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
//0 diffuse, 1 GGX, 2 coat, 3 sheen
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


//strategy pmf by id (0 diffuse, 1 GGX, 2 coat, 3 sheen)
inline float StrategyP(SamplingP p, uint strategy)
{
    if (strategy == 0u) return p.Pdiff;
    if (strategy == 1u) return p.Pspec;
    if (strategy == 2u) return p.Pcoat;
    return p.Psheen;
}

//====================================
//BRDF SAMPLING
//====================================
//ggxNoReflect routes the GGX sampler into the refract branch, NRC owns the reflection delta
//_WithStrategy takes the lobe as INPUT (no selection draw): the lobe-indexed
//replay forces the reservoir's recorded lobe; its callers burn the selection
//draw themselves so the direction dims stay stream-aligned with generation.
inline float3 SampleBRDF_WithStrategy(uint strategy, uint matID, float3 o, float3 n_s, float3 n_g, float3 localKd, half localPr, half localPm, inout uint seed, half etai, half etat, bool ggxNoReflect = false) {
    float3 sample;

    bool refract = false;
    const bool canRefract = true;

    if(strategy == 0){
        sample = SampleBRDF_Lambertian(matID, o, n_s, n_g, seed);
    }
    else if(strategy == 1){
        sample = SampleBRDF_GGX(matID, o, n_s, n_g, etai, etat, refract, seed, localKd, localPr, localPm, canRefract, ggxNoReflect);
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

    //reject below surface samples
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

//strategy-reporting variant: the lobe-indexed PSS records the sampled lobe id
inline float3 SampleBRDF(SamplingP p, uint matID, float3 o, float3 n_s, float3 n_g, float3 localKd, half localPr, half localPm, inout uint seed, half etai, half etat, bool ggxNoReflect, out uint strategyOut) {
    strategyOut = SelectSamplingStrategy(p, seed);
    return SampleBRDF_WithStrategy(strategyOut, matID, o, n_s, n_g, localKd, localPr, localPm, seed, etai, etat, ggxNoReflect);
}

inline float3 SampleBRDF(SamplingP p, uint matID, float3 o, float3 n_s, float3 n_g, float3 localKd, half localPr, half localPm, inout uint seed, half etai, half etat, bool ggxNoReflect = false) {
    uint strategy;
    return SampleBRDF(p, matID, o, n_s, n_g, localKd, localPr, localPm, seed, etai, etat, ggxNoReflect, strategy);
}

//forced-lobe sampling (hybrid replay, RC_F_LOBES): burn the selection draw for
//stream alignment, then sample the SUPPLIED lobe — the direction dims consume
//exactly the draws generation consumed within that lobe.
inline float3 SampleBRDF_Forced(uint strategy, uint matID, float3 o, float3 n_s, float3 n_g, float3 localKd, half localPr, half localPm, inout uint seed, half etai, half etat, bool ggxNoReflect = false) {
    RandomFloatSingle(seed);   //the SelectSamplingStrategy draw
    return SampleBRDF_WithStrategy(strategy, matID, o, n_s, n_g, localKd, localPr, localPm, seed, etai, etat, ggxNoReflect);
}


//====================================
//COMBINED BRDF DATA
//====================================
struct BrdfData {
    float3 val;
    float pdf;
};

//pdf only path, mirrors EvaluateAndPdf_COMBINED's pdf math
//pdf accumulator stays float -- pdf products can dip below fp16 min normal.
inline float BRDF_PDF_COMBINED(
    SamplingP p,
    uint matID, float3 n_s, float3 n_g, float3 s, float3 o,
    float3 localKd, half localPr, half localPm, half etai, half etat)
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
//each lobe skips eval, pdf, transmittance when p.X < EPSILON, SER keeps branches coherent
inline float3 EvaluateBRDF_COMBINED(
    SamplingP p,
    uint matID, float3 n_s, float3 n_g, float3 s, float3 o,
    float3 localKd, half localPr, half localPm, half etai, half etat)
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
//ggxNoReflect must match the value passed to SampleBRDF at the same vertex.
//BrdfData.pdf stays float (pdf products can underflow fp16); val stays float3
//for full radiance range.
inline BrdfData EvaluateAndPdf_COMBINED(
    SamplingP p,
    uint matID, float3 n_s, float3 n_g, float3 s, float3 o,
    float3 localKd, half localPr, half localPm, half etai, half etat,
    bool ggxNoReflect = false)
{
    BrdfData res;
    res.val = 0.0f;
    res.pdf = 0.0f;

    const float3 N  = normalize(n_s);
    const float3 fN = normalize(n_g);
    const float3 V  = normalize(o);
    const float3 L  = normalize(s);

    //gate is a product of transmittances, always in [0,1] — half precision is exact enough
    half gate = (half)1.0;

    if (p.Psheen >= EPSILON) {
        res.val += (float)gate * EvaluateBRDF_SHEEN(matID, n_s, -s, o);
        res.pdf += p.Psheen * BRDF_PDF_SHEEN(matID, n_s, -s, o);
        gate    *= (half)Transmittance_SHEEN(matID, n_s, -s, o);
    }
    if (p.Pcoat >= EPSILON) {
        const CoatResult cr = EvalCoatAll(matID, N, V, L, etai, etat);
        res.val += (float)gate * cr.f;
        res.pdf += p.Pcoat * cr.pdf;
        gate    *= (half)cr.t;
    }
    if (p.Pspec >= EPSILON) {
        const GGXResult gr = EvalGGXAll(matID, N, fN, V, L, etai, etat, localKd, localPr, localPm, ggxNoReflect);
        res.val += (float)gate * gr.f;
        res.pdf += p.Pspec * gr.pdf;
        gate    *= (half)gr.t;
    }
    if (p.Pdiff >= EPSILON) {
        res.val += (float)gate * EvaluateBRDF_Lambertian(matID, n_s, n_g, -s, o, etai, etat, localKd);
        res.pdf += p.Pdiff * BRDF_PDF_Lambertian(matID, n_s, n_g, -s, o);
    }
    return res;
}

//====================================
//SINGLE-LOBE EVAL AND PDF  (lobe-indexed PSS, supp §1)
//====================================
//The gated value of ONE lobe (upper-layer transmittances applied, so the sum
//over lobes equals the COMBINED value) and its OWN directional pdf, NOT
//probability-weighted: {rho_l, p(w|l)}. The p.X EPSILON skip set must match
//EvaluateAndPdf_COMBINED so generation, reconnection and replay agree on what
//a lobe is. Returns {0,0} when the lobe is absent at this vertex (strategy
//probability < EPSILON) — the shift-undefined signal for forced replay.
//Early-outs at the target lobe: layers BELOW it never evaluate.
inline BrdfData EvaluateLobePdf_COMBINED(
    SamplingP p, uint strategy,
    uint matID, float3 n_s, float3 n_g, float3 s, float3 o,
    float3 localKd, half localPr, half localPm, half etai, half etat,
    bool ggxNoReflect = false)
{
    BrdfData res;
    res.val = 0.0f;
    res.pdf = 0.0f;

    const float3 N  = normalize(n_s);
    const float3 fN = normalize(n_g);
    const float3 V  = normalize(o);
    const float3 L  = normalize(s);

    half gate = (half)1.0;

    if (p.Psheen >= EPSILON) {
        if (strategy == 3u) {
            res.val = (float)gate * EvaluateBRDF_SHEEN(matID, n_s, -s, o);
            res.pdf = BRDF_PDF_SHEEN(matID, n_s, -s, o);
            return res;
        }
        gate *= (half)Transmittance_SHEEN(matID, n_s, -s, o);
    }
    if (p.Pcoat >= EPSILON) {
        const CoatResult cr = EvalCoatAll(matID, N, V, L, etai, etat);
        if (strategy == 2u) {
            res.val = (float)gate * cr.f;
            res.pdf = cr.pdf;
            return res;
        }
        gate *= (half)cr.t;
    }
    if (p.Pspec >= EPSILON) {
        const GGXResult gr = EvalGGXAll(matID, N, fN, V, L, etai, etat, localKd, localPr, localPm, ggxNoReflect);
        if (strategy == 1u) {
            res.val = (float)gate * gr.f;
            res.pdf = gr.pdf;
            return res;
        }
        gate *= (half)gr.t;
    }
    if (p.Pdiff >= EPSILON && strategy == 0u) {
        res.val = (float)gate * EvaluateBRDF_Lambertian(matID, n_s, n_g, -s, o, etai, etat, localKd);
        res.pdf = BRDF_PDF_Lambertian(matID, n_s, n_g, -s, o);
    }
    return res;
}

//fused marginal + selected-lobe latch: ONE walk yields the MIS/criteria pdf
//(marginal over lobes, supp §3) AND the sampled lobe's {rho_l, p(w|l)} for the
//lobe-indexed throughput and jacobian bundles. The latched pair matches
//EvaluateLobePdf_COMBINED exactly (same formulas, same skips) so raygen's
//base-side bundle and the reuse-side re-evaluation cancel correctly.
inline BrdfData EvaluateAndPdf_COMBINED_L(
    SamplingP p, uint strategy,
    uint matID, float3 n_s, float3 n_g, float3 s, float3 o,
    float3 localKd, half localPr, half localPm, half etai, half etat,
    bool ggxNoReflect,
    out float3 lobeVal, out float lobePdf)
{
    BrdfData res;
    res.val = 0.0f;
    res.pdf = 0.0f;
    lobeVal = 0.0f;
    lobePdf = 0.0f;

    const float3 N  = normalize(n_s);
    const float3 fN = normalize(n_g);
    const float3 V  = normalize(o);
    const float3 L  = normalize(s);

    half gate = (half)1.0;

    if (p.Psheen >= EPSILON) {
        const float3 fS = EvaluateBRDF_SHEEN(matID, n_s, -s, o);
        const float  pS = BRDF_PDF_SHEEN(matID, n_s, -s, o);
        res.val += (float)gate * fS;
        res.pdf += p.Psheen * pS;
        if (strategy == 3u) { lobeVal = (float)gate * fS; lobePdf = pS; }
        gate    *= (half)Transmittance_SHEEN(matID, n_s, -s, o);
    }
    if (p.Pcoat >= EPSILON) {
        const CoatResult cr = EvalCoatAll(matID, N, V, L, etai, etat);
        res.val += (float)gate * cr.f;
        res.pdf += p.Pcoat * cr.pdf;
        if (strategy == 2u) { lobeVal = (float)gate * cr.f; lobePdf = cr.pdf; }
        gate    *= (half)cr.t;
    }
    if (p.Pspec >= EPSILON) {
        const GGXResult gr = EvalGGXAll(matID, N, fN, V, L, etai, etat, localKd, localPr, localPm, ggxNoReflect);
        res.val += (float)gate * gr.f;
        res.pdf += p.Pspec * gr.pdf;
        if (strategy == 1u) { lobeVal = (float)gate * gr.f; lobePdf = gr.pdf; }
        gate    *= (half)gr.t;
    }
    if (p.Pdiff >= EPSILON) {
        const float3 fD = EvaluateBRDF_Lambertian(matID, n_s, n_g, -s, o, etai, etat, localKd);
        const float  pD = BRDF_PDF_Lambertian(matID, n_s, n_g, -s, o);
        res.val += (float)gate * fD;
        res.pdf += p.Pdiff * pD;
        if (strategy == 0u) { lobeVal = (float)gate * fD; lobePdf = pD; }
    }
    return res;
}


//====================================
//X1 SHARP REFLECTION SPLIT INTEGRAL
//====================================
//peel delta GGX/coat off the primary BSDF, the explicit reflection ray and NRC carry the tail

inline bool ShouldDropDeltaGGX(float Pr, float Pm)
{
    return (Pr < SMOOTH_SPECULAR_THRESHOLD) && (Pm < 0.5f);
}

inline bool IsSmoothTransmissive(uint matID, float Pr)
{
    return (Pr < SMOOTH_SPECULAR_THRESHOLD) && (LoadKd_w(matID) < (1.0f - EPSILON));
}

inline bool ShouldDropDeltaCoat(uint matID)
{
    return (LoadPc(matID) > 0.0f) && (LoadPcr(matID) < SMOOTH_SPECULAR_THRESHOLD);
}

//drops the requested lobes and renormalises, must use the same sp for sample and eval
inline SamplingP DropDeltaLobes(SamplingP sp, bool dropGGX, bool dropCoat)
{
    if (dropGGX)  sp.Pspec = 0.0f;
    if (dropCoat) sp.Pcoat = 0.0f;

    float total = sp.Psheen + sp.Pcoat + sp.Pspec + sp.Pdiff;
    if (total > 0.0f) {
        const float inv = 1.0f / total;
        sp.Psheen *= inv;
        sp.Pcoat  *= inv;
        sp.Pspec  *= inv;
        sp.Pdiff  *= inv;
    } else {
        sp.Psheen = 0.0f; sp.Pcoat = 0.0f; sp.Pspec = 0.0f; sp.Pdiff = 1.0f;
    }
    return sp;
}

//(ComputeSharpReflectionFresnel removed — sole consumer was the dead x1
// sharp-reflection feature.)
