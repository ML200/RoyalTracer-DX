// Walter 2007 microfacet transmission.
inline bool RefractVector(float3 wo, float3 m, float eta, out float3 wi)
{
    float cosWoM  = dot(wo, m);
    float sin2WoM = max(0.0f, 1.0f - cosWoM * cosWoM);
    float k       = 1.0f - eta * eta * sin2WoM;
    if (k <= 0.0f) return false;

    float cosWtM = sqrt(k);

    wi = -eta * wo + (eta * cosWoM - cosWtM) * m;

    float len2 = dot(wi, wi);
    if (len2 <= 1e-16f) return false;
    wi = wi * rsqrt(len2);
    return true;
}

inline float3 MirrorAcrossPlane(float3 v, float3 n)
{
    // Thin glass reflects without refraction.
    return v - 2.0f * dot(v, n) * n;
}

static const float GGX_REFLECT_PICK_MIN = 0.125f;
// Choose reflection probability from Fresnel and lobe availability.
inline float GGXReflectPick(uint mID, float p_refl, float p_tran)
{
    const float p_sum = p_refl + p_tran;
    if (!(p_sum > 0.0f)) return 0.0f;
    const float physical = p_refl / p_sum;
    const bool  transmits = LoadKd_w(mID) < 1.0f - EPSILON;
    return transmits ? max(physical, GGX_REFLECT_PICK_MIN) : physical;
}

inline float3 EvaluateBRDF_GGX(
    uint mID, float3 normal, float3 flatNormal,
    float3 incoming, float3 outgoing,
    float etai, float etat, float3 Kd, float Pr, float Pm)
{
    float3 N = normalize(normal);
    float3 fN = normalize(flatNormal);
    float3 V = normalize(outgoing);
    float3 L = normalize(-incoming);

    float NdotV = abs(dot(N, V)) + 0.00001f;
    float NdotL = dot(N, L);

    bool isReflect = NdotL > 0.0f;

    float r = Pr;
    float alpha = max(0.001f, r * r);
    float metalness = Pm;

    float aniso    = LoadAniso(mID);
    float anisoRot = LoadAnisoRot(mID);
    float ax, ay;
    ComputeAnisotropicAlphas(alpha, aniso, ax, ay);
    float3 T, B;
    BuildAnisotropicFrame(N, anisoRot, T, B);

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
    float3 N    = normalize(normal);
    float3 wo   = normalize(outgoing);
    float gate = LoadKd_w(mID) * (1.0f - Pm);
    return gate * (1.0f - GGXDirectionalReflectance(Pr, max(dot(N, wo), 0.0f), etai, etat));
}

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

// Sample reflection or transmission from the Walter microfacet model.
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
    bool canRefract,
    bool noReflect = false)
{
    float  r         = Pr;
    float  alpha     = max(0.001f, r * r);
    float  metalness = Pm;
    float  trans_w   = 1.0f - LoadKd_w(mID);

    float3 V  = normalize(outgoing);
    float3 N  = normalize(normal);
    float3 fN = normalize(flatNormal);

    float aniso    = LoadAniso(mID);
    float anisoRot = LoadAnisoRot(mID);
    float ax, ay;
    ComputeAnisotropicAlphas(alpha, aniso, ax, ay);
    float3 T, B;
    BuildAnisotropicFrame(N, anisoRot, T, B);

    float3 H;
    if (r < SMOOTH_SPECULAR_THRESHOLD)
        H = N;
    else
        H = SampleVNDF_H_Aniso(ax, ay, V, N, T, B, seed);
    float   VdotH = max(EPSILON, dot(V, H));

    // Keep Fresnel selection in float precision.
    float  F_diel    = FresnelDielectricTIR(V, H, etai, etat).x;
    float  p_refl_H  = (1.0f - metalness) * F_diel + metalness;
    float  p_tran_H  = (1.0f - metalness) * (1.0f - F_diel) * trans_w;
    float  pick_refl = noReflect ? 0.0f : GGXReflectPick(mID, p_refl_H, p_tran_H);
    float  r_pick    = RandomFloatSingle(seed);

    float3 L;
    if (r_pick < pick_refl || !canRefract)
    {
        L = reflect(-V, H);
        refract = false;
    }
    else if (LoadIsThinGlass(mID))
    {

        L = MirrorAcrossPlane(reflect(-V, H), N);
        refract = true;
    }
    else
    {
        float eta = etai / etat;
        if (!RefractVector(V, H, eta, L)) {
            refract = false;
            L = reflect(-V, H);
        }
        else refract = true;
    }

    return normalize(L);
}

inline float GGXTransmittance(
    uint matID, float3 N, float3 V, float3 L,
    half etai, half etat, float3 Kd, half Pr, half Pm)
{
    const float gate = LoadKd_w(matID) * (1.0f - (float)Pm);
    return gate * (1.0f - GGXDirectionalReflectance((float)Pr, max(dot(N, V), 0.0f), etai, etat));
}
struct GGXResult {
    float3 f;
    float  pdf;
    float  t;
};

// Evaluate GGX radiance response for the chosen transport mode.
inline GGXResult EvalGGXAll(
    uint matID, float3 N, float3 fN, float3 V, float3 L,
    half etai, half etat, float3 Kd, half Pr, half Pm,
    bool noReflect = false, bool needTransmission = true)
{
    GGXResult r;
    r.f = 0.0f;
    r.pdf = 0.0f;
    r.t = 1.0f;

    const float NdotV   = abs(dot(N, V)) + 0.00001f;
    const float NdotL_f = dot(N, L);
    const bool  isReflect = NdotL_f > 0.0f;
    const float absNdotL  = abs(NdotL_f);

    const bool  thinTransmit = !isReflect && LoadIsThinGlass(matID);

    const half Kd_w_h     = (half)LoadKd_w(matID);
    const half oneMinusPm = (half)1.0 - Pm;
    const half trans_w    = (half)1.0 - Kd_w_h;

    [branch] if (needTransmission)
    {
        r.t = GGXTransmittance(matID, N, V, L, etai, etat, Kd, Pr, Pm);
    }

    const half alpha = max((half)0.001, Pr * Pr);

    half ax_h, ay_h;
    {
        const float aniso  = LoadAniso(matID);
        const half  aspect = (half)sqrt(1.0f - 0.9f * abs(aniso));
        ax_h = max((half)0.001, alpha / aspect);
        ay_h = max((half)0.001, alpha * aspect);
    }
    float3 T, B;
    BuildAnisotropicFrame(N, LoadAnisoRot(matID), T, B);

    [branch]
    if (thinTransmit)
    {
        const float3 Lm  = MirrorAcrossPlane(L, N);
        const float3 Hun = V + Lm;
        if (dot(Hun, Hun) <= 1e-16f) return r;
        const float3 Ht  = normalize(Hun);

        const float VdotHt = dot(V, Ht);
        const half  NdotHt = (half)dot(N, Ht);
        const half  TdotHt = (half)dot(T, Ht);
        const half  BdotHt = (half)dot(B, Ht);
        const half  TdotV  = (half)dot(T, V);
        const half  BdotV  = (half)dot(B, V);
        const half  TdotLm = (half)dot(T, Lm);
        const half  BdotLm = (half)dot(B, Lm);

        const float D   = D_GGX_Aniso(NdotHt, TdotHt, BdotHt, ax_h, ay_h);
        const float G1V = G1_SmithGGX_Aniso(NdotV,    TdotV,  BdotV,  ax_h, ay_h);
        const float G1L = G1_SmithGGX_Aniso(absNdotL, TdotLm, BdotLm, ax_h, ay_h);
        const float G2  = G1V * G1L;

        const float F  = FresnelDielectricTIR(V, Ht, etai, etat).x;
        const float wT = (1.0f - F) * (float)trans_w * (float)oneMinusPm;

        const float  DG2_over_den = (D * G2) / (4.0f * NdotV * absNdotL);
        float3 spec_t = (wT * DG2_over_den) * LoadTf(matID);
        r.f = (any(isnan(spec_t)) || any(isinf(spec_t))) ? 0.0.xxx : spec_t;

        const float p_refl = (float)oneMinusPm * F + (float)Pm;
        const float p_tran = (float)oneMinusPm * (1.0f - F) * (float)trans_w;
        const float p_sum  = p_refl + p_tran;
        if (p_sum > 0.0f)
        {
            const float VdotHt_pos = max(1e-6f, VdotHt);
            const float pdf_H      = (D * G1V * VdotHt_pos) / max(1e-6f, NdotV);
            const float pTranScale = noReflect ? 1.0f : (1.0f - GGXReflectPick(matID, p_refl, p_tran));
            r.pdf = max(0.0f, pTranScale * pdf_H / (4.0f * VdotHt_pos));
        }
        return r;
    }

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

    const float VdotH = dot(V, H);
    const float LdotH = dot(L, H);

    const half NdotL = (half)NdotL_f;
    const half NdotH = (half)dot(N, H);
    const half TdotH = (half)dot(T, H);
    const half BdotH = (half)dot(B, H);
    const half TdotV = (half)dot(T, V);
    const half BdotV = (half)dot(B, V);
    const half TdotL = (half)dot(T, L);
    const half BdotL = (half)dot(B, L);

    const float D   = D_GGX_Aniso(NdotH, TdotH, BdotH, ax_h, ay_h);
    const float G1V = G1_SmithGGX_Aniso(NdotV,    TdotV, BdotV, ax_h, ay_h);
    const float G1L = G1_SmithGGX_Aniso(absNdotL, TdotL, BdotL, ax_h, ay_h);
    const float G2  = G1V * G1L;

    // Keep Fresnel selection and PDFs in float precision.
    const float3 F_d_vec = FresnelDielectricTIR(V, H, etai, etat);
    const float  F_diel  = F_d_vec.x;

    const float p_refl_H = (float)oneMinusPm * F_diel + (float)Pm;
    const float p_tran_H = (float)oneMinusPm * (1.0f - F_diel) * (float)trans_w;
    const float p_sum    = p_refl_H + p_tran_H;

    if (isReflect)
    {
        if (noReflect)
        {

            r.f = 0.0.xxx;
        }
        else
        {

            const float DG2_over_den = (D * G2) / (4.0f * NdotV * NdotL);
            const float Ess          = GetEssLUT((float)Pr, NdotV);
            const float kms          = (1.0f - Ess) / max(Ess, 1e-6f);

            float3 spec;

            {
                const float3 F0_d = ComputeF0Dielectric(etai, etat);
                spec = ((float)1.0 - (float)Pm) * (F_d_vec * DG2_over_den) * ((float3)1.0 + F0_d * kms);
            }

            {
                const float3 F_c = FresnelConductor(Kd, V, H);
                spec += (float)Pm * (F_c * DG2_over_den) * ((float3)1.0 + Kd * kms);
            }
            r.f = (any(isnan(spec)) || any(isinf(spec))) ? 0.0.xxx : spec;
        }
    }
    else
    {
        float oneMinusF  = 1.0f - F_diel;
        float denom_bsdf = NdotV * absNdotL;
        float denom_jac  = etai * VdotH + etat * LdotH;
        float numer      = (etat * etat) * abs(VdotH) * abs(LdotH);
        float btdf       = oneMinusF * D * G2 * numer / (denom_bsdf * denom_jac * denom_jac);
        float gate_eval  = (float)(trans_w * oneMinusPm);
        float scalar_t   = btdf * gate_eval;
        float3 spec_t    = max(0.0f, float3(scalar_t, scalar_t, scalar_t));
        r.f = (any(isnan(spec_t)) || any(isinf(spec_t))) ? 0.0.xxx : spec_t;
    }

    if (p_sum > 0.0f)
    {
        if (isReflect)
        {
            if (noReflect)
            {
                r.pdf = 0.0f;
            }
            else
            {
                float VdotH_pos = max(1e-6f, VdotH);
                float pdf_H     = (D * G1V * VdotH_pos) / max(1e-6f, NdotV);

                float eta    = etai / etat;
                float cos2_t = 1.0f - (eta * eta) * (1.0f - VdotH_pos * VdotH_pos);
                const float pickRefl = cos2_t < 0.0f ? 1.0f : GGXReflectPick(matID, p_refl_H, p_tran_H);

                r.pdf = max(0.0f, pickRefl * pdf_H / (4.0f * VdotH_pos));
            }
        }
        else
        {
            float VdotH_pos = max(EPSILON, VdotH);
            float pdf_H     = (D * G1V * VdotH_pos) / max(NdotV, EPSILON);

            float denom_jac = etai * VdotH + etat * LdotH;
            float jacobian  = (etat * etat * abs(LdotH)) / (denom_jac * denom_jac);

            const float pTranScale = noReflect ? 1.0f : (1.0f - GGXReflectPick(matID, p_refl_H, p_tran_H));
            r.pdf = max(0.0f, pTranScale * pdf_H * jacobian);
        }
    }

    return r;
}

// Evaluate the selected GGX lobe PDF in solid-angle measure.
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

    if (!reflect && LoadIsThinGlass(mID))
    {
        const float3 Lm = MirrorAcrossPlane(L, N);
        float3 Hun = V + Lm;
        if (dot(Hun, Hun) <= 1e-16f) return 0.0f;
        float3 Ht = normalize(Hun);

        float alpha_t = max(0.001f, Pr * Pr);
        float ax_t, ay_t;
        ComputeAnisotropicAlphas(alpha_t, LoadAniso(mID), ax_t, ay_t);
        float3 Tt, Bt;
        BuildAnisotropicFrame(N, LoadAnisoRot(mID), Tt, Bt);

        float VdotHt = max(1e-6f, dot(V, Ht));
        float pdf_H  = (D_GGX_Aniso(dot(N, Ht), dot(Tt, Ht), dot(Bt, Ht), ax_t, ay_t)
                      * G1_SmithGGX_Aniso(NdotV, dot(Tt, V), dot(Bt, V), ax_t, ay_t) * VdotHt)
                      / max(1e-6f, NdotV);

        float F        = FresnelDielectricTIR(V, Ht, etai, etat).x;
        float trans_wt = 1.0f - LoadKd_w(mID);
        float p_refl   = (1.0f - Pm) * F + Pm;
        float p_tran   = (1.0f - Pm) * (1.0f - F) * trans_wt;
        float p_sum    = p_refl + p_tran;
        if (p_sum <= 0.0f) return 0.0f;
        return max(0.0f, (1.0f - GGXReflectPick(mID, p_refl, p_tran)) * pdf_H / (4.0f * VdotHt));
    }

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

        float eta = etai / etat;
        float cos2_t = 1.0f - (eta*eta) * (1.0f - VdotH_pos*VdotH_pos);
        const float pickRefl = cos2_t < 0.0f ? 1.0f : GGXReflectPick(mID, p_refl_H, p_tran_H);

        return max(0.0f, pickRefl * pdf_H / (4.0f * VdotH_pos));
    }
    else
    {
        float VdotH = dot(V, H);
        float LdotH = dot(L, H);

        float F_diel   = FresnelDielectricTIR(V, H, etai, etat).x;
        float p_refl_H = (1.0f - metalness) * F_diel + metalness;
        float p_tran_H = (1.0f - metalness) * (1.0f - F_diel) * trans_w;

        // Walter 2007 transmission Jacobian.
        float denom = etai * VdotH + etat * LdotH;
        float VdotH_pos = max(EPSILON, VdotH);
        float pdf_H     = (D_GGX_Aniso(NdotH, TdotH, BdotH, ax, ay) * G1_SmithGGX_Aniso(max(NdotV, EPSILON), TdotV, BdotV, ax, ay) * VdotH_pos)
                        / max(NdotV, EPSILON);

        float jacobian = (etat * etat * abs(LdotH)) / (denom * denom);

        return max(0.0f, (1.0f - GGXReflectPick(mID, p_refl_H, p_tran_H)) * pdf_H * jacobian);
    }
}
