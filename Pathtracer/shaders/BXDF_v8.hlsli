struct SamplingP{
    float Psheen;
    float Pcoat;
    float Pspec;
    float Pdiff;
};

static const float STRATEGY_REFLECT_MIN = 0.5f;
static const float STRATEGY_REFLECT_MIN_TRANSMISSIVE = 0.125f;
static const float STRATEGY_REFLECT_GAIN = 4.0f;
inline SamplingP CalculateStrategyProbabilities(uint mID, float3 outgoing, float3 normal, half etai, half etat, float3 Kd, half Pr, half Pm)
{
    // Keep low Fresnel weights in float.
    const float r_sheen = Sampling_Weight_SHEEN(mID, normal, outgoing);
    const float r_coat  = Sampling_Weight_COAT(mID, normal, outgoing, etai, etat);
    const float r_ggx   = Sampling_Weight_GGX(mID, normal, outgoing, etai, etat, Kd, Pm);
    const float r_lamb  = Sampling_Weight_Lambertian(mID, normal, outgoing);

    const float energy_after_sheen = 1.0f - r_sheen;
    const float energy_after_coat  = energy_after_sheen * (1.0f - r_coat);
    const float energy_after_ggx   = energy_after_coat  * (1.0f - r_ggx);

    SamplingP sp;
    sp.Psheen = r_sheen;
    sp.Pcoat  = energy_after_sheen * r_coat * 3.0f;
    sp.Pspec  = energy_after_coat  * r_ggx  * 5.0f;
    sp.Pdiff  = energy_after_ggx   * r_lamb;

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

    const bool  transmissive = LoadKd_w(mID) < 1.0f - EPSILON;
    const bool  ggxReflects  = !transmissive && (float)Pr < BROAD_GGX_ROUGHNESS;
    const float reflection   = sp.Pcoat + (ggxReflects ? sp.Pspec : 0.0f);
    const float rest         = 1.0f - reflection;
    const float floorR       = min(transmissive ? STRATEGY_REFLECT_MIN_TRANSMISSIVE : STRATEGY_REFLECT_MIN,
                                   STRATEGY_REFLECT_GAIN * reflection);
    if (reflection > 0.0f && reflection < floorR && rest > 0.0f)
    {
        const float up   = floorR / reflection;
        const float down = (1.0f - floorR) / rest;
        sp.Pcoat  *= up;
        sp.Pspec  *= ggxReflects ? up : down;
        sp.Pdiff  *= down;
        sp.Psheen *= down;
    }
    return sp;
}

#define LOBE_BROAD 4u
inline bool IsBroadGGX(half Pr) { return (float)Pr >= BROAD_GGX_ROUGHNESS; }

inline bool HasBroadShare(SamplingP p, half Pr, half Pm)
{
    return (p.Pdiff >= EPSILON && (float)Pm < 1.0f - EPSILON) ||
           (p.Pspec >= EPSILON && IsBroadGGX(Pr));
}

inline uint SelectSamplingStrategyFrom(SamplingP p, float r)
{
    float c = p.Pdiff;
    if (r < c) return 0;
    c += p.Pspec;
    if (r < c) return 1;
    c += p.Pcoat;
    if (r < c) return 2;
    return 3;
}
inline uint SelectSamplingStrategy(SamplingP p, inout uint seed)
{
    return SelectSamplingStrategyFrom(p, RandomFloatSingle(seed));
}

inline float StrategyP(SamplingP p, uint strategy)
{
    if (strategy == 0u) return p.Pdiff;
    if (strategy == 1u) return p.Pspec;
    if (strategy == 2u) return p.Pcoat;
    return p.Psheen;
}

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

    // Reject samples on the wrong geometric side.
    const float Ng_wi = dot(sample, n_g);
    if ((!refract && Ng_wi <= 0.0f) || (refract && Ng_wi >= 0.0f))
        return 0.0f;

    return sample;
}

inline float3 SampleBRDF_LambertianFrom(float3 n_s, float3 n_g, float2 u)
{
    const float3 sample = CosineUnitVectorInHemisphereFrom(n_s, u);
    return dot(sample, n_g) <= 0.0f ? 0.0f : sample;
}

inline float3 SampleBRDF(SamplingP p, uint matID, float3 o, float3 n_s, float3 n_g, float3 localKd, half localPr, half localPm, inout uint seed, half etai, half etat, bool ggxNoReflect, out uint strategyOut) {
    strategyOut = SelectSamplingStrategy(p, seed);
    return SampleBRDF_WithStrategy(strategyOut, matID, o, n_s, n_g, localKd, localPr, localPm, seed, etai, etat, ggxNoReflect);
}

inline float3 SampleBRDF(SamplingP p, uint matID, float3 o, float3 n_s, float3 n_g, float3 localKd, half localPr, half localPm, inout uint seed, half etai, half etat, bool ggxNoReflect = false) {
    uint strategy;
    return SampleBRDF(p, matID, o, n_s, n_g, localKd, localPr, localPm, seed, etai, etat, ggxNoReflect, strategy);
}

// Keep PDF accumulation in float.
struct BrdfData {
    float3 val;
    float pdf;
};

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

    half gate = (half)1.0;

    if (p.Psheen >= EPSILON) {
        res.val += (float)gate * EvaluateBRDF_SHEEN(matID, n_s, -s, o);
        res.pdf += p.Psheen * BRDF_PDF_SHEEN(matID, n_s, -s, o);
        gate    *= (half)Transmittance_SHEEN(matID, n_s, -s, o);
    }
    if (p.Pcoat >= EPSILON) {
        const CoatResult cr = EvalCoatAll(matID, N, V, L, etai, etat, p.Pspec >= EPSILON || p.Pdiff >= EPSILON);
        res.val += (float)gate * cr.f;
        res.pdf += p.Pcoat * cr.pdf;
        gate    *= (half)cr.t;
    }
    if (p.Pspec >= EPSILON) {
        const GGXResult gr = EvalGGXAll(matID, N, fN, V, L, etai, etat, localKd, localPr, localPm, ggxNoReflect, p.Pdiff >= EPSILON);
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
        const CoatResult cr = EvalCoatAll(matID, N, V, L, etai, etat, p.Pspec >= EPSILON || p.Pdiff >= EPSILON);
        if (strategy == 2u) {
            res.val = (float)gate * cr.f;
            res.pdf = cr.pdf;
            return res;
        }
        gate *= (half)cr.t;
    }

    float broadPdf = 0.0f, broadP = 0.0f;
    if (p.Pspec >= EPSILON) {
        const GGXResult gr = EvalGGXAll(matID, N, fN, V, L, etai, etat, localKd, localPr, localPm, ggxNoReflect, p.Pdiff >= EPSILON);
        if (strategy == 1u) {
            res.val = (float)gate * gr.f;
            res.pdf = gr.pdf;
            return res;
        }
        if (strategy == LOBE_BROAD && IsBroadGGX(localPr)) {
            res.val += (float)gate * gr.f;
            broadPdf += p.Pspec * gr.pdf; broadP += p.Pspec;
        }
        gate *= (half)gr.t;
    }
    if (p.Pdiff >= EPSILON && (strategy == 0u || strategy == LOBE_BROAD)) {
        const float pD = BRDF_PDF_Lambertian(matID, n_s, n_g, -s, o);
        res.val += (float)gate * EvaluateBRDF_Lambertian(matID, n_s, n_g, -s, o, etai, etat, localKd);
        res.pdf  = pD;
        broadPdf += p.Pdiff * pD; broadP += p.Pdiff;
    }
    if (strategy == LOBE_BROAD) res.pdf = broadP > 0.0f ? broadPdf / broadP : 0.0f;
    return res;
}

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
        const CoatResult cr = EvalCoatAll(matID, N, V, L, etai, etat, p.Pspec >= EPSILON || p.Pdiff >= EPSILON);
        res.val += (float)gate * cr.f;
        res.pdf += p.Pcoat * cr.pdf;
        if (strategy == 2u) { lobeVal = (float)gate * cr.f; lobePdf = cr.pdf; }
        gate    *= (half)cr.t;
    }

    float broadPdf = 0.0f, broadP = 0.0f;
    if (p.Pspec >= EPSILON) {
        const GGXResult gr = EvalGGXAll(matID, N, fN, V, L, etai, etat, localKd, localPr, localPm, ggxNoReflect, p.Pdiff >= EPSILON);
        res.val += (float)gate * gr.f;
        res.pdf += p.Pspec * gr.pdf;
        if (strategy == 1u) { lobeVal = (float)gate * gr.f; lobePdf = gr.pdf; }
        if (strategy == LOBE_BROAD && IsBroadGGX(localPr)) {
            lobeVal += (float)gate * gr.f;
            broadPdf += p.Pspec * gr.pdf; broadP += p.Pspec;
        }
        gate    *= (half)gr.t;
    }
    if (p.Pdiff >= EPSILON) {
        const float3 fD = EvaluateBRDF_Lambertian(matID, n_s, n_g, -s, o, etai, etat, localKd);
        const float  pD = BRDF_PDF_Lambertian(matID, n_s, n_g, -s, o);
        res.val += (float)gate * fD;
        res.pdf += p.Pdiff * pD;
        if (strategy == 0u) { lobeVal = (float)gate * fD; lobePdf = pD; }
        if (strategy == LOBE_BROAD) {
            lobeVal += (float)gate * fD;
            broadPdf += p.Pdiff * pD; broadP += p.Pdiff;
        }
    }
    if (strategy == LOBE_BROAD) lobePdf = broadP > 0.0f ? broadPdf / broadP : 0.0f;
    return res;
}

// One-lobe estimation; diffuse and a broad GGX form one group.
#define LOBE_GROUP_BROAD 0u   // diffuse, with the GGX lobe when IsBroadGGX
#define LOBE_GROUP_SPEC  1u   // GGX
#define LOBE_GROUP_COAT  2u
#define LOBE_GROUP_SHEEN 3u

inline uint LobeGroupOf(uint strategy, half Pr)
{
    return (strategy == 1u && IsBroadGGX(Pr)) ? LOBE_GROUP_BROAD : strategy;
}

inline float LobeGroupP(SamplingP p, uint group, half Pr)
{
    if (group == LOBE_GROUP_BROAD) return p.Pdiff + (IsBroadGGX(Pr) ? p.Pspec : 0.0f);
    if (group == LOBE_GROUP_SPEC)  return p.Pspec;
    if (group == LOBE_GROUP_COAT)  return p.Pcoat;
    return p.Psheen;
}

// Broad group pdf: Pdiff/Pspec mixture, normalized to the group.
inline BrdfData EvaluateLobe(
    SamplingP p, uint group,
    uint matID, float3 n_s, float3 n_g, float3 s, float3 o,
    float3 localKd, half localPr, half localPm, half etai, half etat,
    bool ggxNoReflect = false)
{
    BrdfData res;
    res.val = 0.0f;
    res.pdf = 0.0f;

    if (group == LOBE_GROUP_SHEEN)
    {
        if (p.Psheen >= EPSILON)
        {
            res.val = EvaluateBRDF_SHEEN(matID, n_s, -s, o);
            res.pdf = BRDF_PDF_SHEEN(matID, n_s, -s, o);
        }
        return res;
    }

    const float3 N  = normalize(n_s);
    const float3 fN = normalize(n_g);
    const float3 V  = normalize(o);
    const float3 L  = normalize(s);

    half gate = (half)1.0;
    if (p.Psheen >= EPSILON) gate *= (half)Transmittance_SHEEN(matID, n_s, -s, o);

    if (group == LOBE_GROUP_COAT)
    {
        if (p.Pcoat >= EPSILON)
        {
            const CoatResult cr = EvalCoatAll(matID, N, V, L, etai, etat, false);
            res.val = (float)gate * cr.f;
            res.pdf = cr.pdf;
        }
        return res;
    }
    if (p.Pcoat >= EPSILON) gate *= (half)CoatTransmittance(matID, N, V, L, etai, etat);

    // GGX: the spec group, or the broad group's upper layer.
    float pdfSum = 0.0f, pSum = 0.0f;
    if (p.Pspec >= EPSILON)
    {
        if (group == LOBE_GROUP_SPEC || IsBroadGGX(localPr))
        {
            const GGXResult gr = EvalGGXAll(matID, N, fN, V, L, etai, etat, localKd, localPr, localPm,
                ggxNoReflect, group == LOBE_GROUP_BROAD && p.Pdiff >= EPSILON);
            res.val = (float)gate * gr.f;
            if (group == LOBE_GROUP_SPEC)
            {
                res.pdf = gr.pdf;
                return res;
            }
            pdfSum  = p.Pspec * gr.pdf;
            pSum    = p.Pspec;
            gate   *= (half)gr.t;
        }
        else if (p.Pdiff >= EPSILON)
            gate *= (half)GGXTransmittance(matID, N, V, L, etai, etat, localKd, localPr, localPm);
    }
    else if (group == LOBE_GROUP_SPEC)
        return res;
    if (p.Pdiff >= EPSILON)
    {
        res.val += (float)gate * EvaluateBRDF_Lambertian(matID, n_s, n_g, -s, o, etai, etat, localKd);
        pdfSum  += p.Pdiff * BRDF_PDF_Lambertian(matID, n_s, n_g, -s, o);
        pSum    += p.Pdiff;
    }
    res.pdf = pSum > 0.0f ? pdfSum / pSum : 0.0f;
    return res;
}

inline bool DropBroadLobes(inout SamplingP sp, half Pr)
{
    sp.Pdiff = 0.0f;
    if (IsBroadGGX(Pr)) sp.Pspec = 0.0f;
    const float total = sp.Psheen + sp.Pcoat + sp.Pspec;
    if (total < EPSILON) {
        sp.Psheen = 0.0f; sp.Pcoat = 0.0f; sp.Pspec = 0.0f;
        return false;
    }
    const float inv = 1.0f / total;
    sp.Psheen *= inv;
    sp.Pcoat  *= inv;
    sp.Pspec  *= inv;
    return true;
}
