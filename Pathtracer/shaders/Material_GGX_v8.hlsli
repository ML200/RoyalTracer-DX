// Walter 2007 microfacet transmission mapping
// wo points out of the *current* medium. m is a visible microfacet (dot(wo,m) > 0).
// eta = etai/etat for the current interface.
// Returns false on TIR or degenerate output.
inline bool RefractVector(float3 wo, float3 m, float eta, out float3 wi)
{
    float cosWoM  = dot(wo, m);                       // VNDF ensures > 0
    float sin2WoM = max(0.0f, 1.0f - cosWoM * cosWoM);
    float k       = 1.0f - eta * eta * sin2WoM;       // = cos^2(theta_t)
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

// Evaluate the GGX BRDF/BTDF (specular only)
inline float3 EvaluateBRDF_GGX(
    uint mID, float3 normal, float3 flatNormal,
    float3 incoming, float3 outgoing,
    float etai, float etat)
{
    float3 N = normalize(normal);
    float3 fN = normalize(flatNormal);
    float3 V = normalize(outgoing);    // wo
    float3 L = normalize(-incoming);   // wi

    float NdotV = max(0.0f, dot(N, V));
    float NdotL = dot(N, L);
    float fNdotV = max(0.0f, dot(fN, V));
    float fNdotL = dot(fN, L);

    //if (NdotV <= 0.0f || fNdotV <= 0.0f) return 0.0.xxx;

    bool reflect = NdotL > 0.0f;
    //if (reflect && fNdotL <= 0.0f) return 0.0.xxx;
    //if (!reflect && fNdotL >= 0.0f) return 0.0.xxx;

    float r = materials[mID].Pr_Pm_Ps_Pc.x;
    float alpha = max(0.001f, r * r);
    float metalness = materials[mID].Pr_Pm_Ps_Pc.y;

    // --- Half-vector ---
    float3 H;
    if (reflect) {
        float3 Hun = V + L;
        //if (dot(Hun,Hun) <= 1e-16f) return 0.0.xxx;
        H = normalize(Hun);
    } else {
        // Correct transmission half-vector: H ∝ V + eta * L, then flip to make V·H > 0
        float etaVL = etai / etat;            // V is on the etai side (N·V > 0 above)
        float3 Hun = V + etaVL * L;
        //if (dot(Hun,Hun) <= 1e-16f) return 0.0.xxx;
        H = normalize(Hun);
        if (dot(V, H) < 0.0f) H = -H;         // ensure visible microfacet
    }

    float NdotH = dot(N, H);
    float VdotH = dot(V, H);
    float LdotH = dot(L, H);

    float D = D_GGX(NdotH, alpha);
    float G = G2_SmithGGX(NdotV, abs(NdotL), alpha);

    if (reflect)
    {
        float3 F0_d = ComputeF0Dielectric(etai, etat);
        float3 F0_c = materials[mID].Kd.xyz;
        float3 F_d  = FresnelDielectricTIR(V, H, etai, etat);
        float3 F_c  = FresnelConductor(F0_c, V, H);

        float denom = 4.0f * NdotV * max(0.0f, NdotL);
        if (denom <= 0.0f) return 0.0.xxx;

        float3 specular_d = (F_d * D * G) / denom;
        float3 specular_c = (F_c * D * G) / denom;

        float Ess = ESS_LUT(mID, NdotV);
        float kms = (1.0f - Ess) / max(Ess, 1e-6f);

        float3 spec = (1.0f - metalness) * specular_d * (1.0f + F0_d * kms)
                    + metalness          * specular_c * (1.0f + F0_c * kms);
        return (any(isnan(spec)) || any(isinf(spec))) ? 0.0.xxx : spec;
    }
    else
    {
        // Signed transmission constraints: V·H > 0, L·H < 0
        if (VdotH <= 0.0f || LdotH >= 0.0f) return 0.0.xxx;

        // Scalar Fresnel; short-circuit when transmission has zero weight (incl. TIR → F==1)
        float F = clamp(FresnelDielectricTIR(V, H, etai, etat).x, 0.0f, 1.0f);
        float oneMinusF = 1.0f - F;
        if (oneMinusF <= 0.0f) return 0.0.xxx; // avoid 0 * inf → NaN

        float denom_bsdf = NdotV * abs(NdotL);
        if (abs(denom_bsdf) <= EPSILON) return 0.0.xxx;

        float denom_jac = etai * VdotH + etat * LdotH; // signed
        if (abs(denom_jac) <= EPSILON) return 0.0.xxx;

        // Walter transmission form: needs |V·H| * |L·H|
        float numer = (etat * etat) * abs(VdotH) * abs(LdotH);

        float btdf = (oneMinusF) * D * G * numer / (denom_bsdf * denom_jac * denom_jac);

        // Opaqueness gate (Kd.w==1 → zero) and no transmission for metals
        float gate = (1.0f - materials[mID].Kd.w) * (1.0f - metalness);
        float scalar_t = btdf * gate;

        float3 spec_t = float3(scalar_t, scalar_t, scalar_t);
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
    float  etat)
{
    float  F0   = ComputeF0Dielectric(etai, etat).x;
    float  Favg = F0 + (1.0f - F0) * (1.0f / 21.0f);

    float  Pr   = materials[mID].Pr_Pm_Ps_Pc.x;

    float3 N    = normalize(normal);
    float3 wo   = normalize(outgoing);
    float3 wi   = normalize(-incoming);

    float  Fo = FresnelDielectric(wo, N, etat, etai).x * (1.0f - Pr * 0.8f);
    float  Fi = FresnelDielectric(wi, N, etai, etat).x * (1.0f - Pr * 0.8f);

    float  Kd_frac = Luma(materials[mID].Kd.xyz * materials[mID].Kd.w);

    // No transmission for metals
    float  metalness = materials[mID].Pr_Pm_Ps_Pc.y;
    float  gate      = materials[mID].Kd.w * (1.0f - metalness);

    return gate * (1.0f - Fo) * (1.0f - Fi) * (1.0f / max(1.0f - Kd_frac * Favg, 1e-4f));
}

// Approximate sampling weight for choosing transmission in MIS
inline float Sampling_Weight_GGX(
    uint   mID,
    float3 normal,
    float3 outgoing,
    float  etai,
    float  etat)
{
    float3 N  = normalize(normal);
    float3 wo = normalize(outgoing);

    float  metalness = materials[mID].Pr_Pm_Ps_Pc.y;
    float3 F_c       = FresnelConductor(materials[mID].Kd.xyz, wo, N);
    float  F_d       = FresnelDielectric(wo, N, etat, etai).x;

    return (1.0f - metalness) * F_d + metalness * Luma(F_c);
}

// VNDF sampling GGX (specular reflection + transmission)
inline float3 SampleBRDF_GGX(
    uint   mID,
    float3 outgoing,
    float3 normal,
    float3 flatNormal,
    float  etai,
    float  etat,
    inout uint2 seed)
{
    float  r         = materials[mID].Pr_Pm_Ps_Pc.x;
    float  alpha     = max(0.001f, r * r);
    float  metalness = materials[mID].Pr_Pm_Ps_Pc.y;
    float  trans_w   = 1.0f - materials[mID].Kd.w; // Kd.w == 1 → opaque

    float3 V  = normalize(outgoing);
    float3 N  = normalize(normal);
    float3 fN = normalize(flatNormal);

    //if (dot(N, V) <= 0.0f || dot(fN, V) <= 0.0f) return (float3)0;

    // 1) Sample visible normal H
    float3 H = SampleVNDF_H(alpha, V, N, seed);
    float   VdotH = max(EPSILON, dot(V, H));

    // 2) Path probabilities for this H
    float  F_diel    = FresnelDielectricTIR(V, H, etai, etat);
    float  p_refl_H  = (1.0f - metalness) * F_diel + metalness;
    float  p_tran_H  = (1.0f - metalness) * (1.0f - F_diel) * trans_w;
    float  p_sum     = p_refl_H + p_tran_H;
    //if (p_sum <= 0.0f) return (float3)0;
    float  pick_refl = p_refl_H / p_sum;

    float3 L;
    if (RandomFloat(seed) < pick_refl)
    {
        // Reflection
        L = reflect(-V, H);
        //if (dot(N, L) <= 0.0f || dot(fN, L) <= 0.0f) return (float3)0;
    }
    else
    {
        // Transmission
        float eta = etai / etat; // if you support two-sided IOR, choose by side
        if (!RefractVector(V, H, eta, L)) {
            // TIR for this H → reflect instead (matches PDF logic that adds TIR mass)
            L = reflect(-V, H);
        }
    }

    return normalize(L);
}

// PDF matching SampleBRDF_GGX
inline float BRDF_PDF_GGX(
    uint mID, float3 N, float3 fN,
    float3 wi, float3 wo,
    float etai, float etat)
{
    float3 V = normalize(wo);
    float3 L = normalize(-wi);
    float  NdotV = dot(N, V);
    float  NdotL = dot(N, L);

    /*if (NdotV <= 0.0f ||
       (NdotL > 0.0f && dot(fN, L) <= 0.0f) ||
       (NdotL < 0.0f && dot(fN, L) >= 0.0f))
        return 0.0f;*/

    bool reflect = NdotL > 0.0f;

    float3 H;
    if (reflect) {
        float3 Hun = V + L;
        //if (dot(Hun,Hun) <= 1e-16f) return 0.0f;
        H = normalize(Hun);
    } else {
        float etaVL = etai / etat;            // V is on etai side here
        float3 Hun = V + etaVL * L;
        //if (dot(Hun,Hun) <= 1e-16f) return 0.0f;
        H = normalize(Hun);
        if (dot(V, H) < 0.0f) H = -H;         // visible microfacet
    }

    float NdotH = dot(N, H);
    //if (NdotH <= 0.0f) return 0.0f;

    float  r         = materials[mID].Pr_Pm_Ps_Pc.x;
    float  alpha     = max(0.001f, r * r);
    float  metalness = materials[mID].Pr_Pm_Ps_Pc.y;
    float  trans_w   = 1.0f - materials[mID].Kd.w;

    float VdotH_abs = max(1e-6f, abs(dot(V, H)));
    float pdf_H     = (D_GGX(NdotH, alpha) * G1_SmithGGX(NdotV, alpha) * VdotH_abs) / max(1e-6f, NdotV);

    float F_diel    = FresnelDielectricTIR(V, H, etai, etat);
    float p_refl_H  = (1.0f - metalness) * F_diel + metalness;
    float p_tran_H  = (1.0f - metalness) * (1.0f - F_diel) * trans_w;
    float p_sum     = p_refl_H + p_tran_H;
    //if (p_sum <= 0.0f) return 0.0f;

    if (reflect)
    {
        float VdotH_pos = max(1e-6f, dot(V, H));      // reflection requires V·H > 0
        float p_sel     = p_refl_H;

        // Add TIR mass if transmit would be invalid for this H
        float eta = etai / etat;
        float cos2_t = 1.0f - (eta*eta) * (1.0f - VdotH_pos*VdotH_pos);
        if (cos2_t < 0.0f) p_sel += p_tran_H;

        return (p_sel / p_sum) * pdf_H / (4.0f * VdotH_pos);
    }
    else
    {
        float VdotH = dot(V, H); // > 0 after flip
        float LdotH = dot(L, H); // < 0 for transmission
        //if (VdotH <= 0.0f || LdotH >= 0.0f) return 0.0f;

        // Fresnel + mixing weights first, so we can short-circuit when transmission has no mass.
        float F_diel   = clamp(FresnelDielectricTIR(V, H, etai, etat).x, 0.0f, 1.0f);
        float p_refl_H = (1.0f - metalness) * F_diel + metalness;
        float p_tran_H = (1.0f - metalness) * (1.0f - F_diel) * trans_w;
        float p_sum    = p_refl_H + p_tran_H;
        //if (p_sum <= EPSILON) return 0.0f;
        //if (p_tran_H <= EPSILON) return 0.0f; // avoid 0 * inf → NaN

        // Signed denominator for refraction mapping (Walter 2007)
        float denom = etai * VdotH + etat * LdotH; // signed
        //if (abs(denom) <= EPSILON) return 0.0f;

        // VNDF pdf for H (visible normals), use V·H (not abs) and protect tiny divisors
        float VdotH_pos = max(EPSILON, VdotH);
        float pdf_H     = (D_GGX(NdotH, alpha) * G1_SmithGGX(max(NdotV, EPSILON), alpha) * VdotH_pos)
                        / max(NdotV, EPSILON);

        // Jacobian for refraction mapping (only |L·H|)
        float jacobian = (etat * etat * abs(LdotH)) / (denom * denom);
        //if (!(jacobian > 0.0f)) return 0.0f;

        return (p_tran_H / p_sum) * pdf_H * jacobian;
    }
}

