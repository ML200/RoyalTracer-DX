//====================================================================
//REFRACTION VECTOR
//====================================================================
//Walter 2007 microfacet transmission mapping
inline bool RefractVector(float3 wo, float3 m, float eta, out float3 wi)
{
    float cosWoM  = dot(wo, m);
    float sin2WoM = max(0.0f, 1.0f - cosWoM * cosWoM);
    float k       = 1.0f - eta * eta * sin2WoM;
    if (k <= 0.0f) return false;                      //TIR

    float cosWtM = sqrt(k);
    //Walter et al. Eq. (10)
    wi = -eta * wo + (eta * cosWoM - cosWtM) * m;

    float len2 = dot(wi, wi);
    if (len2 <= 1e-16f) return false;
    wi = wi * rsqrt(len2);
    return true;
}

//====================================================================
//GGX BRDF EVALUATION
//====================================================================
inline float3 EvaluateBRDF_GGX(
    uint mID, float3 normal, float3 flatNormal,
    float3 incoming, float3 outgoing,
    float etai, float etat, float3 Kd, float Pr, float Pm)
{
    float3 N = normalize(normal);
    float3 fN = normalize(flatNormal);
    float3 V = normalize(outgoing);    //wo
    float3 L = normalize(-incoming);   //wi

    float NdotV = abs(dot(N, V)) + 0.00001f;
    float NdotL = dot(N, L);

    bool isReflect = NdotL > 0.0f;

    float r = Pr;
    float alpha = max(0.001f, r * r);
    float metalness = Pm;

    //Anisotropy setup
    float aniso    = LoadAniso(mID);
    float anisoRot = LoadAnisoRot(mID);
    float ax, ay;
    ComputeAnisotropicAlphas(alpha, aniso, ax, ay);
    float3 T, B;
    BuildAnisotropicFrame(N, anisoRot, T, B);

    //Half-vector
    float3 H;
    if (isReflect) {
        float3 Hun = V + L;
        if (dot(Hun, Hun) <= 1e-16f) return 0.0.xxx;
        H = normalize(Hun);
    } else {
        float3 Hun = etai * V + etat * L;
        if (dot(Hun, Hun) <= 1e-16f) return 0.0.xxx;
        H = normalize(Hun);
        if (dot(V, H) < 0.0f) H = -H;
    }

    float NdotH = dot(N, H);
    float VdotH = dot(V, H);
    float LdotH = dot(L, H);

    //Anisotropic NDF and masking-shadowing
    float TdotH = dot(T, H);
    float BdotH = dot(B, H);
    float TdotV = dot(T, V);
    float BdotV = dot(B, V);
    float TdotL = dot(T, L);
    float BdotL = dot(B, L);

    float D = D_GGX_Aniso(NdotH, TdotH, BdotH, ax, ay);
    float G = G2_SmithGGX_Aniso(NdotV, TdotV, BdotV, abs(NdotL), TdotL, BdotL, ax, ay);

    if (isReflect)
    {
        float3 F0_d = ComputeF0Dielectric(etai, etat);
        float3 F0_c = Kd;
        float3 F_d  = FresnelDielectricTIR(V, H, etai, etat);
        float3 F_c  = FresnelConductor(F0_c, V, H);

        float denom = 4.0f * NdotV * NdotL;

        float3 specular_d = (F_d * D * G) / denom;
        float3 specular_c = (F_c * D * G) / denom;

        float Ess = GetEssLUT(Pr, NdotV);
        float kms = (1.0f - Ess) / max(Ess, 1e-6f);

        float3 spec = (1.0f - metalness) * specular_d * (1.0f + F0_d * kms)
                    + metalness          * specular_c * (1.0f + F0_c * kms);
        return (any(isnan(spec)) || any(isinf(spec))) ? 0.0.xxx : spec;
    }
    else
    {
        float F = FresnelDielectricTIR(V, H, etai, etat).x;
        float oneMinusF = 1.0f - F;

        float denom_bsdf = NdotV * abs(NdotL);
        float denom_jac = etai * VdotH + etat * LdotH;

        float numer = (etat * etat) * abs(VdotH) * abs(LdotH);

        float btdf = (oneMinusF) * D * G * numer / (denom_bsdf * denom_jac * denom_jac);
        float gate = (1.0f - LoadKd_w(mID)) * (1.0f - metalness);
        float scalar_t = btdf * gate;

        float3 spec_t = max(0.0f, float3(scalar_t, scalar_t, scalar_t));
        return (any(isnan(spec_t)) || any(isinf(spec_t))) ? 0.0.xxx : spec_t;
    }
}


//====================================================================
//GGX TRANSMITTANCE
//====================================================================
//Fraction of energy that goes into transmission for dielectrics
inline float Transmittance_GGX(
    uint   mID,
    float3 normal,
    float3 incoming,
    float3 outgoing,
    float  etai,
    float  etat,
    float3 Kd,
    float Pr,
    float Pm)
{
    float  Ni   = LoadNi(mID);
    float  F0   = ComputeF0Dielectric(etat, etai).x;
    float  Favg = (F0 + (1.0f - F0) * (1.0f / 21.0f)) * (1.0f/Ni);

    float3 N    = normalize(normal);
    float3 wo   = normalize(outgoing);
    float3 wi   = normalize(-incoming);

    float  Fo = FresnelDielectric(wo, N, etat, etai).x * (1.0f - Pr * 0.7f) * (1.0f - Pr * 0.7f);
    float  Fi = FresnelDielectric(wi, N, etai, etat).x;

    float  Kd_frac = Avg3(Kd * LoadKd_w(mID));

    //no transmission for metals
    float  metalness = Pm;
    float  gate      = LoadKd_w(mID) * (1.0f - metalness);

    return gate * (1.0f - Fo) * (1.0f - Fi) * (1.0f / max(1.0f - Kd_frac * Favg, 1e-4f));
}


//====================================================================
//GGX SAMPLING WEIGHT
//====================================================================
//Approximate sampling weight for choosing transmission in MIS
inline float Sampling_Weight_GGX(
    uint   mID,
    float3 normal,
    float3 outgoing,
    float  etai,
    float  etat,
    float3 Kd,
    float Pm)
{
    float3 N  = normalize(normal);
    float3 wo = normalize(outgoing);

    float  metalness = Pm;
    float3 F_c       = FresnelConductor(Kd, wo, N);
    float  F_d       = FresnelDielectricTIR(wo, N, etai, etat).x;

    return (1.0f - metalness) * F_d + metalness * Luma(F_c) + (1 - F_d) * (1.0 - LoadKd_w(mID));
}


//====================================================================
//GGX VNDF SAMPLING
//====================================================================
inline float3 SampleBRDF_GGX(
    uint   mID,
    float3 outgoing,
    float3 normal,
    float3 flatNormal,
    float  etai,
    float  etat,
    inout bool refract,
    inout uint seed,
    float3 Kd,
    float Pr,
    float Pm,
    bool canRefract)
{
    float  r         = Pr;
    float  alpha     = max(0.001f, r * r);
    float  metalness = Pm;
    float  trans_w   = 1.0f - LoadKd_w(mID);

    float3 V  = normalize(outgoing);
    float3 N  = normalize(normal);
    float3 fN = normalize(flatNormal);

    //Anisotropy setup
    float aniso    = LoadAniso(mID);
    float anisoRot = LoadAnisoRot(mID);
    float ax, ay;
    ComputeAnisotropicAlphas(alpha, aniso, ax, ay);
    float3 T, B;
    BuildAnisotropicFrame(N, anisoRot, T, B);

    //Sample visible normal, fall back to perfect specular for very smooth surfaces
    float3 H;
    if (r < SMOOTH_SPECULAR_THRESHOLD)
        H = N;
    else
        H = SampleVNDF_H_Aniso(ax, ay, V, N, T, B, seed);
    float   VdotH = max(EPSILON, dot(V, H));

    //Reflection and transmission probabilities
    float  F_diel    = FresnelDielectricTIR(V, H, etai, etat).x;
    float  p_refl_H  = (1.0f - metalness) * F_diel + metalness;
    float  p_tran_H  = (1.0f - metalness) * (1.0f - F_diel) * trans_w;
    float  p_sum     = p_refl_H + p_tran_H;
    float  pick_refl = p_refl_H / p_sum;

    float3 L;
    if (RandomFloatSingle(seed) < pick_refl || !canRefract)
    {
        //Reflection
        L = reflect(-V, H);
        refract = false;
    }
    else
    {
        //Transmission
        float eta = etai / etat;
        if (!RefractVector(V, H, eta, L)) {
            refract = false;
            L = reflect(-V, H);
        }
        else refract = true;
    }

    return normalize(L);
}


//====================================================================
//FUSED GGX EVAL, PDF, TRANSMITTANCE
//====================================================================
struct GGXResult {
    float3 f;
    float  pdf;
    float  t;
};

inline GGXResult EvalGGXAll(
    uint matID, float3 N, float3 fN, float3 V, float3 L,
    float etai, float etat, float3 Kd, float Pr, float Pm)
{
    GGXResult r;
    r.f = 0.0f;
    r.pdf = 0.0f;

    //Transmittance, independent of half-vector
    float NdotV = abs(dot(N, V)) + 0.00001f;
    float NdotL = dot(N, L);

    const float Kd_w = LoadKd_w(matID);
    {
        float Ni   = LoadNi(matID);
        float F0_t = ComputeF0Dielectric(etat, etai).x;
        float Favg = (F0_t + (1.0f - F0_t) * (1.0f / 21.0f)) * (1.0f / Ni);
        float Fo   = FresnelDielectric(V, N, etat, etai).x * (1.0f - Pr * 0.7f) * (1.0f - Pr * 0.7f);
        float Fi   = FresnelDielectric(L, N, etai, etat).x;
        float Kd_frac = Avg3(Kd * Kd_w);
        float gate_t  = Kd_w * (1.0f - Pm);
        r.t = gate_t * (1.0f - Fo) * (1.0f - Fi) * (1.0f / max(1.0f - Kd_frac * Favg, 1e-4f));
    }

    //Eval and PDF shared setup
    bool  isReflect = NdotL > 0.0f;
    float absNdotL  = abs(NdotL);

    float alpha   = max(0.001f, Pr * Pr);
    float trans_w = 1.0f - Kd_w;

    float ax, ay;
    ComputeAnisotropicAlphas(alpha, LoadAniso(matID), ax, ay);
    float3 T, B;
    BuildAnisotropicFrame(N, LoadAnisoRot(matID), T, B);

    //Half vector
    float3 H;
    if (isReflect) {
        float3 Hun = V + L;
        if (dot(Hun, Hun) <= 1e-16f) return r;
        H = normalize(Hun);
    } else {
        float3 Hun = etai * V + etat * L;
        if (dot(Hun, Hun) <= 1e-16f) return r;
        H = normalize(Hun);
        if (dot(V, H) < 0.0f) H = -H;
    }

    float NdotH = dot(N, H);
    float VdotH = dot(V, H);
    float LdotH = dot(L, H);
    float TdotH = dot(T, H);
    float BdotH = dot(B, H);
    float TdotV = dot(T, V);
    float BdotV = dot(B, V);
    float TdotL = dot(T, L);
    float BdotL = dot(B, L);

    //Microfacet terms
    float D   = D_GGX_Aniso(NdotH, TdotH, BdotH, ax, ay);
    float G1V = G1_SmithGGX_Aniso(NdotV, TdotV, BdotV, ax, ay);
    float G1L = G1_SmithGGX_Aniso(absNdotL, TdotL, BdotL, ax, ay);
    float G2  = G1V * G1L;

    //Fresnel at H
    float3 F_d_vec = FresnelDielectricTIR(V, H, etai, etat);
    float  F_diel  = F_d_vec.x;

    //Selection probabilities
    float p_refl_H = (1.0f - Pm) * F_diel + Pm;
    float p_tran_H = (1.0f - Pm) * (1.0f - F_diel) * trans_w;
    float p_sum    = p_refl_H + p_tran_H;

    //Eval
    if (isReflect)
    {
        float3 F0_d = ComputeF0Dielectric(etai, etat);
        float3 F_c  = FresnelConductor(Kd, V, H);

        float denom = 4.0f * NdotV * NdotL;

        float3 specular_d = (F_d_vec * D * G2) / denom;
        float3 specular_c = (F_c * D * G2) / denom;

        float Ess = GetEssLUT(Pr, NdotV);
        float kms = (1.0f - Ess) / max(Ess, 1e-6f);

        float3 spec = (1.0f - Pm) * specular_d * (1.0f + F0_d * kms)
                    + Pm          * specular_c * (1.0f + Kd  * kms);
        r.f = (any(isnan(spec)) || any(isinf(spec))) ? 0.0.xxx : spec;
    }
    else
    {
        float oneMinusF  = 1.0f - F_diel;
        float denom_bsdf = NdotV * absNdotL;
        float denom_jac  = etai * VdotH + etat * LdotH;
        float numer      = (etat * etat) * abs(VdotH) * abs(LdotH);
        float btdf       = oneMinusF * D * G2 * numer / (denom_bsdf * denom_jac * denom_jac);
        float gate_eval  = trans_w * (1.0f - Pm);
        float scalar_t   = btdf * gate_eval;
        float3 spec_t    = max(0.0f, float3(scalar_t, scalar_t, scalar_t));
        r.f = (any(isnan(spec_t)) || any(isinf(spec_t))) ? 0.0.xxx : spec_t;
    }

    //PDF
    if (p_sum > 0.0f)
    {
        if (isReflect)
        {
            float VdotH_pos = max(1e-6f, VdotH);
            float pdf_H     = (D * G1V * VdotH_pos) / max(1e-6f, NdotV);

            float p_sel = p_refl_H;
            float eta   = etai / etat;
            float cos2_t = 1.0f - (eta * eta) * (1.0f - VdotH_pos * VdotH_pos);
            if (cos2_t < 0.0f) p_sel += p_tran_H;

            r.pdf = max(0.0f, (p_sel / p_sum) * pdf_H / (4.0f * VdotH_pos));
        }
        else
        {
            float VdotH_pos = max(EPSILON, VdotH);
            float pdf_H     = (D * G1V * VdotH_pos) / max(NdotV, EPSILON);

            float denom_jac = etai * VdotH + etat * LdotH;
            float jacobian  = (etat * etat * abs(LdotH)) / (denom_jac * denom_jac);

            r.pdf = max(0.0f, (p_tran_H / p_sum) * pdf_H * jacobian);
        }
    }

    return r;
}


//====================================================================
//GGX PDF
//====================================================================
//Matches SampleBRDF_GGX
inline float BRDF_PDF_GGX(
    uint mID, float3 N, float3 fN,
    float3 wi, float3 wo,
    float etai, float etat, float3 Kd, float Pr, float Pm)
{
    float3 V = normalize(wo);
    float3 L = normalize(-wi);
    float  NdotV = abs(dot(N, V))+0.0001f;
    float  NdotL = dot(N, L);
    float fNdotL = dot(fN, L);

    bool reflect = NdotL > 0.0f;

    float3 H;
    if (reflect) {
        float3 Hun = V + L;
        if (dot(Hun,Hun) <= 1e-16f) return 0.0f;
        H = normalize(Hun);
    } else {
        float etaVL = etai / etat;
        float3 Hun = etai * V + etat * L;
        if (dot(Hun,Hun) <= 1e-16f) return 0.0f;
        H = normalize(Hun);
        if (dot(V, H) < 0.0f) H = -H;
    }

    float NdotH = dot(N, H);

    float  r         = Pr;
    float  alpha     = max(0.001f, r * r);
    float  metalness = Pm;
    float  trans_w   = 1.0f - LoadKd_w(mID);

    //Anisotropy setup
    float aniso    = LoadAniso(mID);
    float anisoRot = LoadAnisoRot(mID);
    float ax, ay;
    ComputeAnisotropicAlphas(alpha, aniso, ax, ay);
    float3 T, B;
    BuildAnisotropicFrame(N, anisoRot, T, B);

    float TdotH = dot(T, H);
    float BdotH = dot(B, H);
    float TdotV = dot(T, V);
    float BdotV = dot(B, V);

    float VdotH_abs = max(1e-6f, abs(dot(V, H)));
    float pdf_H     = (D_GGX_Aniso(NdotH, TdotH, BdotH, ax, ay) * G1_SmithGGX_Aniso(NdotV, TdotV, BdotV, ax, ay) * VdotH_abs) / max(1e-6f, NdotV);

    float F_diel    = FresnelDielectricTIR(V, H, etai, etat).x;
    float p_refl_H  = (1.0f - metalness) * F_diel + metalness;
    float p_tran_H  = (1.0f - metalness) * (1.0f - F_diel) * trans_w;
    float p_sum     = p_refl_H + p_tran_H;
    if (p_sum <= 0.0f) return 0.0f;

    if (reflect)
    {
        float VdotH_pos = max(1e-6f, dot(V, H));
        float p_sel     = p_refl_H;

        //Add TIR mass if transmit would be invalid for this H
        float eta = etai / etat;
        float cos2_t = 1.0f - (eta*eta) * (1.0f - VdotH_pos*VdotH_pos);
        if (cos2_t < 0.0f) p_sel += p_tran_H;

        return max(0.0f, (p_sel / p_sum) * pdf_H / (4.0f * VdotH_pos));
    }
    else
    {
        float VdotH = dot(V, H);
        float LdotH = dot(L, H);

        //Fresnel
        float F_diel   = FresnelDielectricTIR(V, H, etai, etat).x;
        float p_refl_H = (1.0f - metalness) * F_diel + metalness;
        float p_tran_H = (1.0f - metalness) * (1.0f - F_diel) * trans_w;
        float p_sum    = p_refl_H + p_tran_H;

        //Signed denominator for refraction mapping, Walter 2007
        float denom = etai * VdotH + etat * LdotH;
        float VdotH_pos = max(EPSILON, VdotH);
        float pdf_H     = (D_GGX_Aniso(NdotH, TdotH, BdotH, ax, ay) * G1_SmithGGX_Aniso(max(NdotV, EPSILON), TdotV, BdotV, ax, ay) * VdotH_pos)
                        / max(NdotV, EPSILON);

        //Jacobian for refraction mapping
        float jacobian = (etat * etat * abs(LdotH)) / (denom * denom);

        return max(0.0f, (p_tran_H / p_sum) * pdf_H * jacobian);
    }
}

