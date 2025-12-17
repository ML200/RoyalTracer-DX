// Walter 2007 microfacet transmission mapping
inline bool RefractVector(float3 wo, float3 m, float eta, out float3 wi)
{
    float cosWoM  = dot(wo, m);
    float sin2WoM = max(0.0f, 1.0f - cosWoM * cosWoM);
    float k       = 1.0f - eta * eta * sin2WoM;
    if (k <= 0.0f) return false;                      // TIR

    float cosWtM = sqrt(k);
    // Walter et al. Eq. (10)
    wi = -eta * wo + (eta * cosWoM - cosWtM) * m;

    float len2 = dot(wi, wi);
    if (len2 <= 1e-16f) return false;
    wi = wi * rsqrt(len2);
    return true;
}

// -----------------------------------------------------------------------------
// GGX BRDF/BXDF
// -----------------------------------------------------------------------------

// Evaluate the GGX BRDF/BTDF
inline float3 EvaluateBRDF_GGX(
    uint mID, float3 normal, float3 flatNormal,
    float3 incoming, float3 outgoing,
    float etai, float etat, float3 Kd, float Pr, float Pm)
{
    float3 N = normalize(normal);
    float3 fN = normalize(flatNormal);
    float3 V = normalize(outgoing);    // wo
    float3 L = normalize(-incoming);   // wi

    float NdotV = abs(dot(N, V))+0.00001f;
    float NdotL = dot(N, L);
    float fNdotL = dot(fN, L);

    bool reflect = NdotL > 0.0f;

    float r = Pr;
    float alpha = max(0.001f, r * r);
    float metalness = Pm;

    // --- Half-vector ---
    float3 H;
    if (reflect) {
        float3 Hun = V + L;
        if (dot(Hun,Hun) <= 1e-16f) return 0.0.xxx;
        H = normalize(Hun);
    } else {
        // Correct transmission half-vector
        float etaVL = etai / etat;
        float3 Hun = etai * V + etat * L;
        if (dot(Hun,Hun) <= 1e-16f) return 0.0.xxx;
        H = normalize(Hun);
        if (dot(V, H) < 0.0f) H = -H;
    }

    float NdotH = dot(N, H);
    float VdotH = dot(V, H);
    float LdotH = dot(L, H);

    float D = D_GGX(NdotH, alpha);
    float G = G2_SmithGGX(NdotV, abs(NdotL), alpha);

    if (reflect)
    {
        //if(etai == 1.0f) return 0.0f;
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

        // Walter transmission form
        float numer = (etat * etat) * abs(VdotH) * abs(LdotH);

        float btdf = (oneMinusF) * D * G * numer / (denom_bsdf * denom_jac * denom_jac);
        float gate = (1.0f - materials[mID].Kd.w) * (1.0f - metalness);
        float scalar_t = btdf * gate;

        float3 spec_t = max(0.0f, float3(scalar_t, scalar_t, scalar_t));
        return (any(isnan(spec_t)) || any(isinf(spec_t))) ? 0.0.xxx : spec_t;
    }
}


// Fraction of energy that goes into transmission (dielectrics)
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
    float  Ni   = materials[mID].Ni;
    float  F0   = ComputeF0Dielectric(etat, etai).x;
    float  Favg = (F0 + (1.0f - F0) * (1.0f / 21.0f)) * (1.0f/Ni);

    float3 N    = normalize(normal);
    float3 wo   = normalize(outgoing);
    float3 wi   = normalize(-incoming);

    float  Fo = FresnelDielectric(wo, N, etat, etai).x * (1.0f - Pr * 0.7f) * (1.0f - Pr * 0.7f);
    float  Fi = FresnelDielectric(wi, N, etai, etat).x;

    float  Kd_frac = Avg3(Kd * materials[mID].Kd.w);

    // No transmission for metals
    float  metalness = Pm;
    float  gate      = materials[mID].Kd.w * (1.0f - metalness);

    return gate * (1.0f - Fo) * (1.0f - Fi) * (1.0f / max(1.0f - Kd_frac * Favg, 1e-4f));
}


// Approximate sampling weight for choosing transmission in MIS
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

    return (1.0f - metalness) * F_d + metalness * Luma(F_c) + (1 - F_d) * (1.0 - materials[mID].Kd.w);
}


// VNDF sampling GGX
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
    float  trans_w   = 1.0f - materials[mID].Kd.w;

    float3 V  = normalize(outgoing);
    float3 N  = normalize(normal);
    float3 fN = normalize(flatNormal);

    // 1) Sample visible normal H
    float3 H = SampleVNDF_H(alpha, V, N, seed);
    float   VdotH = max(EPSILON, dot(V, H));

    // 2) Path probabilities for this H
    float  F_diel    = FresnelDielectricTIR(V, H, etai, etat).x;
    float  p_refl_H  = (1.0f - metalness) * F_diel + metalness;
    float  p_tran_H  = (1.0f - metalness) * (1.0f - F_diel) * trans_w;
    float  p_sum     = p_refl_H + p_tran_H;
    float  pick_refl = p_refl_H / p_sum;

    float3 L;
    if (RandomFloatSingle(seed) < pick_refl || !canRefract)
    {
        // Reflection
        L = reflect(-V, H);
        refract = false;
        //if (dot(N, L) <= 0.0f || dot(fN, L) <= 0.0f) return (float3)0;
    }
    else
    {
        // Transmission
        float eta = etai / etat;
        if (!RefractVector(V, H, eta, L)) {
            refract = false;
            L = reflect(-V, H);
        }
        else refract = true;
    }

    return normalize(L);
}


// PDF matching SampleBRDF_GGX
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
    float  trans_w   = 1.0f - materials[mID].Kd.w;

    float VdotH_abs = max(1e-6f, abs(dot(V, H)));
    float pdf_H     = (D_GGX(NdotH, alpha) * G1_SmithGGX(NdotV, alpha) * VdotH_abs) / max(1e-6f, NdotV);

    float F_diel    = FresnelDielectricTIR(V, H, etai, etat).x;
    float p_refl_H  = (1.0f - metalness) * F_diel + metalness;
    float p_tran_H  = (1.0f - metalness) * (1.0f - F_diel) * trans_w;
    float p_sum     = p_refl_H + p_tran_H;
    if (p_sum <= 0.0f) return 0.0f;

    if (reflect)
    {
        float VdotH_pos = max(1e-6f, dot(V, H));
        float p_sel     = p_refl_H;

        // Add TIR mass if transmit would be invalid for this H
        float eta = etai / etat;
        float cos2_t = 1.0f - (eta*eta) * (1.0f - VdotH_pos*VdotH_pos);
        if (cos2_t < 0.0f) p_sel += p_tran_H;

        return max(0.0f, (p_sel / p_sum) * pdf_H / (4.0f * VdotH_pos));
    }
    else
    {
        float VdotH = dot(V, H);
        float LdotH = dot(L, H);

        // Fresnel
        float F_diel   = FresnelDielectricTIR(V, H, etai, etat).x;
        float p_refl_H = (1.0f - metalness) * F_diel + metalness;
        float p_tran_H = (1.0f - metalness) * (1.0f - F_diel) * trans_w;
        float p_sum    = p_refl_H + p_tran_H;

        // Signed denominator for refraction mapping (Walter 2007)
        float denom = etai * VdotH + etat * LdotH;
        float VdotH_pos = max(EPSILON, VdotH);
        float pdf_H     = (D_GGX(NdotH, alpha) * G1_SmithGGX(max(NdotV, EPSILON), alpha) * VdotH_pos)
                        / max(NdotV, EPSILON);

        // Jacobian for refraction mapping
        float jacobian = (etat * etat * abs(LdotH)) / (denom * denom);

        return max(0.0f, (p_tran_H / p_sum) * pdf_H * jacobian);
    }
}

