//====================================
//REFRACTION VECTOR
//====================================
//Walter 2007 microfacet transmission
inline bool RefractVector(float3 wo, float3 m, float eta, out float3 wi)
{
    float cosWoM  = dot(wo, m);
    float sin2WoM = max(0.0f, 1.0f - cosWoM * cosWoM);
    float k       = 1.0f - eta * eta * sin2WoM;
    if (k <= 0.0f) return false;

    float cosWtM = sqrt(k);
    //Walter et al. Eq 10
    wi = -eta * wo + (eta * cosWoM - cosWtM) * m;

    float len2 = dot(wi, wi);
    if (len2 <= 1e-16f) return false;
    wi = wi * rsqrt(len2);
    return true;
}

//====================================
//THIN-GLASS STRAIGHT-THROUGH
//====================================
//Mirror a direction across the surface plane (flips only the normal component). Thin-glass
//transmission folds the lower-hemisphere L into this mirror and evaluates the GGX REFLECTION
//lobe against it: at Pr=0 the mirror of reflect(-V,N) is exactly -V (straight through, no Snell
//bend); with roughness it spreads symmetrically to the reflection lobe. Tangential dots are
//mirror-invariant (u·N=0 -> u·mirror(L)=u·L), so only the half-vector and NdotL sign change.
inline float3 MirrorAcrossPlane(float3 v, float3 n)
{
    return v - 2.0f * dot(v, n) * n;
}

//====================================
//GGX BRDF EVALUATION
//====================================
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

    //anisotropy
    float aniso    = LoadAniso(mID);
    float anisoRot = LoadAnisoRot(mID);
    float ax, ay;
    ComputeAnisotropicAlphas(alpha, aniso, ax, ay);
    float3 T, B;
    BuildAnisotropicFrame(N, anisoRot, T, B);

    //half-vector
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


//====================================
//GGX TRANSMITTANCE
//====================================
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


//====================================
//GGX SAMPLING WEIGHT
//====================================
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


//====================================
//GGX VNDF SAMPLING
//====================================
//noReflect routes into the refract branch, NRC supplies the reflection delta
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

    //VNDF sample, perfect specular for very smooth surfaces
    float3 H;
    if (r < SMOOTH_SPECULAR_THRESHOLD)
        H = N;
    else
        H = SampleVNDF_H_Aniso(ax, ay, V, N, T, B, seed);
    float   VdotH = max(EPSILON, dot(V, H));

    //always draw, RNG must advance identically with or without noReflect
    float  F_diel    = FresnelDielectricTIR(V, H, etai, etat).x;
    float  p_refl_H  = (1.0f - metalness) * F_diel + metalness;
    float  p_tran_H  = (1.0f - metalness) * (1.0f - F_diel) * trans_w;
    float  p_sum     = p_refl_H + p_tran_H;
    float  pick_refl = noReflect ? 0.0f : (p_refl_H / p_sum);
    float  r_pick    = RandomFloatSingle(seed);

    float3 L;
    if (r_pick < pick_refl || !canRefract)
    {
        L = reflect(-V, H);
        refract = false;
    }
    else if (LoadIsThinGlass(mID))
    {
        //thin glass: straight-through. Mirror the reflected micro-facet direction across the
        //surface -> -V at Pr=0 (no bend), spreading with roughness. NdotL<0, so refract=true
        //keeps BXDF's below-surface acceptance (the sample stays on the far side).
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


//====================================
//FUSED GGX EVAL PDF TRANSMITTANCE
//====================================
struct GGXResult {
    float3 f;
    float  pdf;
    float  t;
};

//noReflect mirrors SampleBRDF_GGX, reflection zero, transmit pdf drops p_tran_H/p_sum.
//Pr/Pm/etai/etat are half (bounded values <= ~2.5); matches the top-level BXDF API
//so the boundary casts collapse. Kd stays float3 (lambertian path takes float Kd).
inline GGXResult EvalGGXAll(
    uint matID, float3 N, float3 fN, float3 V, float3 L,
    half etai, half etat, float3 Kd, half Pr, half Pm,
    bool noReflect = false)
{
    GGXResult r;
    r.f = 0.0f;
    r.pdf = 0.0f;

    //NdotV stays float, the +1e-5 offset is denormal in fp16 and would flush to zero
    const float NdotV   = abs(dot(N, V)) + 0.00001f;
    const float NdotL_f = dot(N, L);
    const bool  isReflect = NdotL_f > 0.0f;
    const float absNdotL  = abs(NdotL_f);

    //thin-walled glass: the transmit lobe is a mirrored reflection, not a refraction (see below).
    const bool  thinTransmit = !isReflect && LoadIsThinGlass(matID);

    //bounded material params already half from the signature
    const half Kd_w_h     = (half)LoadKd_w(matID);
    const half oneMinusPm = (half)1.0 - Pm;
    const half trans_w    = (half)1.0 - Kd_w_h;

    //transmittance, all factors lie in [0,1]
    {
        const float Ni       = LoadNi(matID);
        const half  F0_t     = (half)ComputeF0Dielectric(etat, etai).x;
        const half  Favg     = (F0_t + ((half)1.0 - F0_t) * (half)(1.0f / 21.0f)) * (half)(1.0f / Ni);
        const half  PrFactor = (half)1.0 - Pr * (half)0.7;
        const half  Fo       = (half)FresnelDielectric(V, N, etat, etai).x * PrFactor * PrFactor;
        const half  Fi       = (half)FresnelDielectric(L, N, etai, etat).x;
        const half  Kd_frac  = (half)Avg3(Kd) * Kd_w_h;
        const half  gate_t   = Kd_w_h * oneMinusPm;
        r.t = (float)(gate_t * ((half)1.0 - Fo) * ((half)1.0 - Fi)
                     / max((half)1.0 - Kd_frac * Favg, (half)1e-4));
    }

    //alpha clamped at 0.001 stays inside fp16 normal range
    const half alpha = max((half)0.001, Pr * Pr);

    //inline ComputeAnisotropicAlphas so ax/ay stay in half registers
    half ax_h, ay_h;
    {
        const float aniso  = LoadAniso(matID);
        const half  aspect = (half)sqrt(1.0f - 0.9f * abs(aniso));
        ax_h = max((half)0.001, alpha / aspect);
        ay_h = max((half)0.001, alpha * aspect);
    }
    float3 T, B;
    BuildAnisotropicFrame(N, LoadAnisoRot(matID), T, B);

    //====================================
    //THIN-GLASS TRANSMIT (mirrored reflection lobe)
    //====================================
    //Evaluate the GGX reflection lobe against the mirror of L, weight by (1-F)*trans_w (no metal
    //transmit) and tint by Tf. Early return keeps the solid-glass refraction path below untouched
    //and off this branch. r.t was already set above (gate for any diffuse layer beneath).
    [branch]
    if (thinTransmit)
    {
        const float3 Lm  = MirrorAcrossPlane(L, N);   //NdotLm = absNdotL > 0
        const float3 Hun = V + Lm;
        if (dot(Hun, Hun) <= 1e-16f) return r;
        const float3 Ht  = normalize(Hun);

        const float VdotHt = dot(V, Ht);
        const half  NdotHt = (half)dot(N, Ht);
        const half  TdotHt = (half)dot(T, Ht);
        const half  BdotHt = (half)dot(B, Ht);
        const half  TdotV  = (half)dot(T, V);
        const half  BdotV  = (half)dot(B, V);
        const half  TdotLm = (half)dot(T, Lm);        //== dot(T,L), mirror-invariant
        const half  BdotLm = (half)dot(B, Lm);

        const float D   = D_GGX_Aniso(NdotHt, TdotHt, BdotHt, ax_h, ay_h);
        const float G1V = G1_SmithGGX_Aniso(NdotV,    TdotV,  BdotV,  ax_h, ay_h);
        const float G1L = G1_SmithGGX_Aniso(absNdotL, TdotLm, BdotLm, ax_h, ay_h);
        const float G2  = G1V * G1L;

        const float F  = FresnelDielectricTIR(V, Ht, etai, etat).x;   //air->glass, real Ni
        const float wT = (1.0f - F) * (float)trans_w * (float)oneMinusPm;

        const float  DG2_over_den = (D * G2) / (4.0f * NdotV * absNdotL);
        float3 spec_t = (wT * DG2_over_den) * LoadTf(matID);
        r.f = (any(isnan(spec_t)) || any(isinf(spec_t))) ? 0.0.xxx : spec_t;

        //pdf: reflection-form half-vector pdf scaled by the transmit selection fraction
        const half p_refl = oneMinusPm * (half)F + Pm;
        const half p_tran = oneMinusPm * ((half)1.0 - (half)F) * trans_w;
        const half p_sum  = p_refl + p_tran;
        if (p_sum > (half)0.0)
        {
            const float VdotHt_pos = max(1e-6f, VdotHt);
            const float pdf_H      = (D * G1V * VdotHt_pos) / max(1e-6f, NdotV);
            const float pTranScale = noReflect ? 1.0f : ((float)p_tran / (float)p_sum);
            r.pdf = max(0.0f, pTranScale * pdf_H / (4.0f * VdotHt_pos));
        }
        return r;
    }

    //half vector
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

    //VdotH/LdotH stay float, used in signed denom_jac with very small values
    const float VdotH = dot(V, H);
    const float LdotH = dot(L, H);

    //frame-relative dots are bounded in [-1,1] and only feed multiplicative D/G terms
    const half NdotL = (half)NdotL_f;
    const half NdotH = (half)dot(N, H);
    const half TdotH = (half)dot(T, H);
    const half BdotH = (half)dot(B, H);
    const half TdotV = (half)dot(T, V);
    const half BdotV = (half)dot(B, V);
    const half TdotL = (half)dot(T, L);
    const half BdotL = (half)dot(B, L);

    //D and G can exceed fp16 range for smooth surfaces, keep float
    const float D   = D_GGX_Aniso(NdotH, TdotH, BdotH, ax_h, ay_h);
    const float G1V = G1_SmithGGX_Aniso(NdotV,    TdotV, BdotV, ax_h, ay_h);
    const float G1L = G1_SmithGGX_Aniso(absNdotL, TdotL, BdotL, ax_h, ay_h);
    const float G2  = G1V * G1L;

    const float3 F_d_vec = FresnelDielectricTIR(V, H, etai, etat);
    const half   F_diel  = (half)F_d_vec.x;

    const half p_refl_H = oneMinusPm * F_diel + Pm;
    const half p_tran_H = oneMinusPm * ((half)1.0 - F_diel) * trans_w;
    const half p_sum    = p_refl_H + p_tran_H;

    //eval
    if (isReflect)
    {
        if (noReflect)
        {
            //reflection branch is owned by the external NRC tap
            r.f = 0.0.xxx;
        }
        else
        {
            //Scoping the dielectric and conductor float3 intermediates into
            //separate blocks keeps the eval peak register count down --
            //F0_d / F_d_vec die before F_c is computed, and F_c / specular_c
            //die before the final combine. Pulls DG2_over_den + kms scalars
            //terrain so they're shared across both sub-blocks instead of recomputed.
            const float DG2_over_den = (D * G2) / (4.0f * NdotV * NdotL);
            const float Ess          = GetEssLUT((float)Pr, NdotV);
            const float kms          = (1.0f - Ess) / max(Ess, 1e-6f);

            float3 spec;
            //dielectric (1-Pm) contribution -- consumes F_d_vec from outer scope
            {
                const float3 F0_d = ComputeF0Dielectric(etai, etat);
                spec = ((float)1.0 - (float)Pm) * (F_d_vec * DG2_over_den) * ((float3)1.0 + F0_d * kms);
            }
            //conductor Pm contribution
            {
                const float3 F_c = FresnelConductor(Kd, V, H);
                spec += (float)Pm * (F_c * DG2_over_den) * ((float3)1.0 + Kd * kms);
            }
            r.f = (any(isnan(spec)) || any(isinf(spec))) ? 0.0.xxx : spec;
        }
    }
    else
    {
        float oneMinusF  = 1.0f - (float)F_diel;
        float denom_bsdf = NdotV * absNdotL;
        float denom_jac  = etai * VdotH + etat * LdotH;
        float numer      = (etat * etat) * abs(VdotH) * abs(LdotH);
        float btdf       = oneMinusF * D * G2 * numer / (denom_bsdf * denom_jac * denom_jac);
        float gate_eval  = (float)(trans_w * oneMinusPm);
        float scalar_t   = btdf * gate_eval;
        float3 spec_t    = max(0.0f, float3(scalar_t, scalar_t, scalar_t));
        r.f = (any(isnan(spec_t)) || any(isinf(spec_t))) ? 0.0.xxx : spec_t;
    }

    //pdf
    if (p_sum > (half)0.0)
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

                half  p_sel  = p_refl_H;
                float eta    = etai / etat;
                float cos2_t = 1.0f - (eta * eta) * (1.0f - VdotH_pos * VdotH_pos);
                if (cos2_t < 0.0f) p_sel += p_tran_H;

                r.pdf = max(0.0f, ((float)p_sel / (float)p_sum) * pdf_H / (4.0f * VdotH_pos));
            }
        }
        else
        {
            float VdotH_pos = max(EPSILON, VdotH);
            float pdf_H     = (D * G1V * VdotH_pos) / max(NdotV, EPSILON);

            float denom_jac = etai * VdotH + etat * LdotH;
            float jacobian  = (etat * etat * abs(LdotH)) / (denom_jac * denom_jac);

            //noReflect collapses pick_tran to 1 — sampler always selects refract
            const float pTranScale = noReflect ? 1.0f : ((float)p_tran_H / (float)p_sum);
            r.pdf = max(0.0f, pTranScale * pdf_H * jacobian);
        }
    }

    return r;
}


//====================================
//GGX PDF
//====================================
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

    //thin-glass transmit: mirrored-reflection pdf (mirrors EvalGGXAll's thinTransmit branch).
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
        return max(0.0f, (p_tran / p_sum) * pdf_H / (4.0f * VdotHt));
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
        float p_sel     = p_refl_H;

        //TIR mass if transmit invalid for this H
        float eta = etai / etat;
        float cos2_t = 1.0f - (eta*eta) * (1.0f - VdotH_pos*VdotH_pos);
        if (cos2_t < 0.0f) p_sel += p_tran_H;

        return max(0.0f, (p_sel / p_sum) * pdf_H / (4.0f * VdotH_pos));
    }
    else
    {
        float VdotH = dot(V, H);
        float LdotH = dot(L, H);

        float F_diel   = FresnelDielectricTIR(V, H, etai, etat).x;
        float p_refl_H = (1.0f - metalness) * F_diel + metalness;
        float p_tran_H = (1.0f - metalness) * (1.0f - F_diel) * trans_w;
        float p_sum    = p_refl_H + p_tran_H;

        //signed denominator, Walter 2007
        float denom = etai * VdotH + etat * LdotH;
        float VdotH_pos = max(EPSILON, VdotH);
        float pdf_H     = (D_GGX_Aniso(NdotH, TdotH, BdotH, ax, ay) * G1_SmithGGX_Aniso(max(NdotV, EPSILON), TdotV, BdotV, ax, ay) * VdotH_pos)
                        / max(NdotV, EPSILON);

        float jacobian = (etat * etat * abs(LdotH)) / (denom * denom);

        return max(0.0f, (p_tran_H / p_sum) * pdf_H * jacobian);
    }
}
