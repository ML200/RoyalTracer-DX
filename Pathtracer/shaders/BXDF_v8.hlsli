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
    const float rough    = materials[mID].Pr_Pm_Ps_Pc.x;
    const float metallic = materials[mID].Pr_Pm_Ps_Pc.y;
    const float sheen_w  = saturate(materials[mID].Pr_Pm_Ps_Pc.z);
    const float clear_w  = saturate(materials[mID].Pr_Pm_Ps_Pc.w);
    const float trans_w  = saturate(1.0f - materials[mID].Kd.w);          // transmission amount
    const float3 baseAlb = materials[mID].Kd.rgb;
    float3 N   = normalize(normal);
    float3 V   = normalize(outgoing);
    float  NoV = saturate(dot(N, V));

    // Base F0 (RGB) and Schlick Fresnel at this view for the base interface
    float3 F0_rgb = ComputeBaseF0(mID);
    float3 Fv_rgb = SchlickFresnel(F0_rgb, NoV);
    float  Rv     = saturate(Avg3(Fv_rgb));
    float  Tv     = 1.0 - Rv;

    // clearcoat (fixed IOR)
    const float F0_coat = 0.04;
    float Fv_coat = pow(1.0 - NoV, 5.0f) * (1.0 - F0_coat) + F0_coat;

    // Everything under the coat is reduced by the coats Fresnel
    float coat_gate = 1.0 - clear_w * Fv_coat;

    // specular
    float w_spec_u  = lerp(Rv, 1.0, metallic) * coat_gate;

    // clearcoat (topmost)
    float w_clear_u = clear_w * Fv_coat;

    // transmission (dielectric only), also under the coat
    float w_trans_u = (1.0 - metallic) * trans_w * Tv * coat_gate;

    // Base (diffuse + sheen) exists for non-metals and only from the non-reflected / non-transmitted share, under the coat
    float base_gate = (1.0 - metallic) * (1.0 - trans_w) * Tv * coat_gate;

    // Diffuse
    float w_diff_u  = base_gate * max(1e-3, Luma(baseAlb));

    // view-side visibility for Charlie
    float G1v_sheen = 1.0f / (1.0f + SHEEN_Lambda_Charlie(NoV));

    // mixture weight for sheen
    float w_sheen_u = base_gate * sheen_w * G1v_sheen;

    // normalize
    float sum = max(1e-6, w_diff_u + w_spec_u + w_clear_u + w_sheen_u + w_trans_u);

    float p_diff  = w_diff_u  / sum;
    float p_spec  = w_spec_u  / sum;
    float p_clear = w_clear_u / sum;
    float p_sheen = w_sheen_u / sum;
    float p_trans = w_trans_u / sum;

    SamplingP sp;
    sp.Psheen = p_sheen;
    sp.Pcoat = p_clear;
    sp.Pspec = p_spec;
    sp.Pdiff = p_diff;
    sp.Ptrans = p_trans;
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
    // Defensive
    r = min(r, 1.0f - EPSILON);

    // CDF
    float c = p.Pspec;                 if (r < c) return 1;  // spec
    c += p.Pcoat;                      if (r < c) return 2;  // clearcoat
    c += p.Psheen;                      if (r < c) return 3;  // sheen
    c += p.Ptrans;                  if (r < c) return 4;  // transmission
    return 0; // diffuse
}


// Sample the BRDF of the given strategy
inline void SampleBRDF(uint mID, float3 outgoing, float3 normal, float3 flatNormal, inout float3 sample, inout RandomData rdata) {
    // Probabilities
    SamplingP p = CalculateStrategyProbabilities(mID, outgoing, normal);
    // Select one method
    uint strategy = SelectSamplingStrategy(mID, outgoing, normal, rdata);

    //Sample from the selected strategy
    if(strategy == 0){ // diffuse
        SampleBRDF_Lambertian(mID, outgoing, normal, flatNormal, sample, rdata.seed);
    }
    else if(strategy == 1){ // specular
        SampleBRDF_GGX(mID, outgoing, normal, flatNormal, sample, rdata.seed);
    }
    else if(strategy == 2){ // coat
        SampleBRDF_COAT(mID, outgoing, normal, flatNormal, sample, rdata.seed);
    }
    else if(strategy == 3){ //sheen
        SampleBRDF_SHEEN(mID, outgoing, normal, flatNormal, sample, rdata.seed);
    }
    else if(strategy == 4){ // transmisson
        SampleBTDF_GGX_TRANS(mID, outgoing, normal, flatNormal, sample, rdata.seed);
    }
    else{ // fallback
        SampleBRDF_Lambertian(mID, outgoing, normal, flatNormal, sample, rdata.seed);
    }
}

// Evaluate all lobes branch-free using the SAME weights
inline float3 EvaluateBRDF_COMBINED(uint mID, float3 N, float3 wi, float3 wo)
{
    float gate = 1.0f;   // cumulative transmission through all layers above
    float T = 1.0f;      // per-layer transmission returned by each Evaluate

    float3 f = 0.0.xxx;

    // Top: SHEEN
    float3 f_sheen = EvaluateBRDF_SHEEN(mID, N, wi, wo, T); // sheen
    f += gate * f_sheen;
    gate *= T;

    // Clear COAT
    float3 f_coat = EvaluateBRDF_COAT(mID, N, wi, wo, T);   // coat
    f += gate * f_coat;
    gate *= T;

    // Base SPECULAR
    float3 f_spec = EvaluateBRDF_GGX(mID, N, wi, wo, T);    // spec
    f += gate * f_spec;
    gate *= T;

    // Base DIFFUSE
    float3 f_diff = EvaluateBRDF_Lambertian(mID, N, wi, wo, T); // diff
    f += gate * f_diff;
    gate *= T;

    //Transmission lobe. It lives at the base, so just use the current gate
    float3 f_btdf = EvaluateBTDF_GGX_TRANS(mID, N, wi, wo, T);
    f += gate * f_btdf;

    return f;
}

inline float BRDF_PDF_COMBINED(uint mID, float3 N, float3 wi, float3 wo)
{
    SamplingP p = CalculateStrategyProbabilities(mID, wo, N);

    float pd = BRDF_PDF_Lambertian(mID, N, wi, wo);
    float ps = BRDF_PDF_GGX(mID, N, wi, wo);
    float pc = BRDF_PDF_COAT(mID, N, wi, wo);
    float ph = BRDF_PDF_SHEEN(mID, N, wi, wo);
    float pt = BTDF_PDF_GGX_TRANS(mID, N, wi, wo);

    return p.Pdiff * pd + p.Pspec * ps + p.Pcoat * pc + p.Psheen * ph + p.Ptrans * pt;
}
