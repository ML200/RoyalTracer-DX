// Evaluate the GGX BRDF for the given material
inline float3 EvaluateBRDF_GGX(
    uint mID,
    float3 normal,
    float3 flatNormal,
    float3 incoming,
    float3 outgoing,
    float etai,
    float etat)
{
    float3 N = normalize(normal);
    float3 fN = normalize(flatNormal);
    float3 V = normalize(outgoing);    // wo
    float3 L = normalize(-incoming);   // wi

    float NdotV = max(0.0f, dot(N, V));
    float NdotL = dot(N, L);
    float fNdotV = max(0.0f, dot(fN, V));
    float fNdotL = dot(fN, L);

    // V must be above both normals
    if (NdotV <= 0.0f || fNdotV <= 0.0f) return 0.0.xxx;

    bool reflect = NdotL > 0.0f;

    // Check geometric validity
    if (reflect && fNdotL <= 0.0f) return 0.0.xxx;  // Reflected ray below geometric surface
    if (!reflect && fNdotL >= 0.0f) return 0.0.xxx; // Refracted ray above geometric surface

    // --- Calculate H ---
    float3 H;
    if (reflect) {
        H = normalize(V + L);
    } else {
        H = normalize(-(etai * V + etat * L));
    }

    float NdotH = max(0.0f, dot(N, H));
    float VdotH = max(0.0f, dot(V, H));
    float LdotH = dot(L, H); // Can be negative

    // --- Get material properties ---
    float alpha = materials[mID].Pr_Pm_Ps_Pc.x;
    alpha *= alpha; // alpha = roughness^2
    alpha = max(0.001f, alpha);
    float metalness = materials[mID].Pr_Pm_Ps_Pc.y;

    // --- Microfacet terms ---
    float D = D_GGX(NdotH, alpha);
    float G = G2_SmithGGX(NdotV, abs(NdotL), alpha);

    if (reflect)
    {
        // --- Fresnel ---
        float3 F0_d = ComputeF0Dielectric(etai, etat);
        float3 F0_c = materials[mID].Kd.xyz;
        float3 F_d  = FresnelDielectricTIR(V, H, etai, etat); // Assuming this returns float3
        float3 F_c  = FresnelConductor(F0_c, V, H);

        float denom = 4.0f * NdotV * NdotL;
        if (denom == 0.0f) return 0.0.xxx;

        // Use your original reflection + multiscatter logic
        float3 specular_d = (F_d * D * G) / denom; // dielectric
        float3 specular_c = (F_c * D * G) / denom; // conductor

        float Ess  = ESS_LUT(mID, NdotV); // Assuming this LUT is for reflection
        float kms  = (1.0f - Ess) / Ess;

        float3 specular_ess = (1.0f-metalness) * specular_d * (1.0f + F0_d * kms)
                           + metalness * specular_c * (1.0f + F0_c * kms);

        return (any(isnan(specular_ess)) || any(isinf(specular_ess))) ? 0.0.xxx : specular_ess;
    }
    else // Transmit
    {
        // Conductors (metals) do not transmit
        if (metalness > 0.0f) return 0.0.xxx;

        // Transmission is only for dielectrics
        float F = FresnelDielectricTIR(V, H, etai, etat); // Scalar Fresnel

        // --- START FIX ---
        // Get the RAW, SIGNED dot products
        float LdotH_raw = dot(L, H);
        float VdotH_raw = dot(V, H);

        // For this H definition, both V.H and L.H must be negative
        if (LdotH_raw >= 0.0f || VdotH_raw >= 0.0f) return 0.0.xxx;

        float denom_bsdf = NdotV * abs(NdotL);
        // Use SIGNED values for the jacobian denominator
        float denom_jac = (etai * VdotH_raw + etat * LdotH_raw);

        if (denom_bsdf == 0.0f || denom_jac == 0.0f) return 0.0.xxx;

        // Use ABSOLUTE values for the jacobian numerator
        float jacobian_term = (etat * etat * abs(LdotH_raw) * abs(VdotH_raw)) / (denom_jac * denom_jac);
        // --- END FIX ---

        float btdf_val = (1.0f - F) * D * G * jacobian_term / denom_bsdf;

        // First, compute the final scalar transmission value
        float scalar_t = btdf_val * materials[mID].Kd.w;

        // Then, explicitly construct the float3 using that scalar for all components
        float3 specular_t = float3(scalar_t, scalar_t, scalar_t);

        return (any(isnan(specular_t)) || any(isinf(specular_t))) ? 0.0.xxx : specular_t;
    }
}

// etaR is the ior of the material the ray is currently in before hitting the material
inline float Transmittance_GGX(uint mID, float3 normal, float3 incoming, float3 outgoing, float etai, float etat){
    // Transmittance is (1-Fo)(1-Fi)(1/(1-Kd*Favg) -> only for dielectrics
    float F0 = ComputeF0Dielectric(etai, etat).x;
    float  Favg    = F0 + (1.0f - F0) * (1.0f/21.0f);

    float Pr = materials[mID].Pr_Pm_Ps_Pc.x;

    // TODO: find a better alternative to the hacky energy conservation
    // Compute the fresnel term for the incoming and outgoing ray
    float F_o = FresnelDielectric(outgoing, normal, etat, etai).x * (1.0f - Pr * 0.8f);
    float F_i = FresnelDielectric(-incoming, normal, etai, etat).x * (1.0f - Pr * 0.8f);

    // Material properties
    float Kd_frac = Luma(materials[mID].Kd.xyz * materials[mID].Kd.w); // Fraction of the Kd layer of the material that might reflect back into the specular layer

    // Metals dont transmit any energy. Energy refracted also wont count towards the diffuse layer
    return  materials[mID].Kd.w * (1.0f - materials[mID].Pr_Pm_Ps_Pc.y) * (1.0f - F_o) * (1.0f - F_i) * (1.0f / (1.0f - Kd_frac * Favg));
}

// Sampling weight function describes an approximation of the transmittance before sampling, as incoming isnt known yet
inline float Sampling_Weight_GGX(uint mID, float3 normal, float3 outgoing, float etai, float etat){
    // We approximate here with a fresnel term coupled with the metalness of the material
    return (1.0f-materials[mID].Pr_Pm_Ps_Pc.y) * FresnelDielectric(outgoing, normal, etat, etai).x + materials[mID].Pr_Pm_Ps_Pc.y * Luma(FresnelConductor(materials[mID].Kd.xyz,outgoing, normal));
}


// VNDF sampling GGX
// Renamed to SampleBSDF_GGX as it handles both reflection and transmission
inline float3 SampleBRDF_GGX(
    uint mID,
    float3  outgoing,
    float3  normal,
    float3  flatNormal, // <-- This is now used correctly
    float   etai,
    float   etat,
    inout uint2 seed)
{
    // --- 1. Setup ---
    float r = materials[mID].Pr_Pm_Ps_Pc.x;
    float alpha = max(0.001f, r * r);
    float metalness = materials[mID].Pr_Pm_Ps_Pc.y;

    float3 V = normalize(outgoing);
    float3 N = normalize(normal);
    float3 fN = normalize(flatNormal); // <-- Get geometric normal

    float NdotV = dot(N, V);
    float fNdotV = dot(fN, V); // <-- Check against geometric normal

    // V must be above *both* normals
    if (NdotV <= 0.0f || fNdotV <= 0.0f) return 0.0.xxx;

    // --- 2. Sample the microfacet normal ---
    float3 H = SampleVNDF_H(alpha, V, N, seed);
    float VdotH = max(EPSILON, dot(V, H));

    // --- 3. Compute Reflection Probability (Fresnel) ---
    float F_diel = FresnelDielectricTIR(V, H, etai, etat);
    float prob_reflect = (1.0f - metalness) * F_diel + metalness * 1.0f;

    float rand = RandomFloat(seed);
    float3 sample_L; // This will be the -incoming direction

    if (rand < prob_reflect)
    {
        // --- 4a. Sample Reflection ---
        sample_L = reflect(-V, H);

        // --- FIX: Validate against BOTH normals ---
        float NdotL = dot(N, sample_L);
        float fNdotL = dot(fN, sample_L);
        if (NdotL <= 0.0f || fNdotL <= 0.0f)
            return 0.0.xxx;
    }
    else
    {
        // --- 4b. Sample Transmission ---
        if (metalness > 0.0f)
            return 0.0.xxx;

        float eta = etai / etat;
        if (!RefractVector(V, H, eta, sample_L))
        {
            // --- 4c. Total Internal Reflection: Fallback to reflection ---
            sample_L = reflect(-V, H);

            // --- FIX: Validate against BOTH normals ---
            float NdotL = dot(N, sample_L);
            float fNdotL = dot(fN, sample_L);
            if (NdotL <= 0.0f || fNdotL <= 0.0f)
                return 0.0.xxx;
        }
        else
        {
            // --- 4d. Successful Refraction ---

            // --- FIX: Validate against BOTH normals ---
            // Refracted ray *must* be *below* both
            float NdotL = dot(N, sample_L);
            float fNdotL = dot(fN, sample_L);
            if (NdotL >= 0.0f || fNdotL >= 0.0f)
                return 0.0.xxx;
        }
    }

    // Return the sampled direction L = -wi
    return normalize(sample_L);
}

// Calculate the PDF for a given sample direction using GGX
// Calculate the PDF for a given sample direction using the mixed BSDF
inline float BRDF_PDF_GGX(
    uint mID,
    float3 N,
    float3 fN,
    float3 wi,
    float3 wo,
    float etai,
    float etat)
{
    float3 V = normalize(wo);
    float3 L = normalize(-wi);
    float NdotV = max(0.0f, dot(N, V));
    float fNdotV = max(0.0f, dot(fN, V));
    float NdotL = dot(N, L);
    float fNdotL = dot(fN, L);

    // V must be above both normals
    if (NdotV <= 0.0f || fNdotV <= 0.0f) return 0.0f;

    bool reflect = NdotL > 0.0f;

    // Check geometric validity
    if (reflect && fNdotL <= 0.0f) return 0.0f;  // Reflected ray below geometric surface
    if (!reflect && fNdotL >= 0.0f) return 0.0f; // Refracted ray above geometric surface

    // --- Calculate H ---
    float3 H;
    if (reflect) {
        H = normalize(V + L);
    } else {
        // H for refraction
        H = normalize(-(etai * V + etat * L));
    }

    // H must be in the upper hemisphere of N
    float NdotH = dot(N, H);
    if (NdotH <= 0.0f) return 0.0f;

    float VdotH = max(EPSILON, dot(V, H));

    // --- Get material properties ---
    float r = materials[mID].Pr_Pm_Ps_Pc.x;
    float alpha = max(0.001f, r * r);
    float metalness = materials[mID].Pr_Pm_Ps_Pc.y;

    // --- Calculate prob_reflect / prob_transmit ---
    // This is the probability of *choosing* reflection/transmission given H
    float F_diel = FresnelDielectricTIR(V, H, etai, etat);
    float prob_reflect = (1.0f - metalness) * F_diel + metalness * 1.0f;
    float prob_transmit = (1.0f - metalness) * (1.0f - F_diel);

    // --- Calculate pdf_H (PDF of sampling H from VNDF) ---
    float D = D_GGX(NdotH, alpha);
    float G1 = G1_SmithGGX(NdotV, alpha);
    float pdf_H = (D * G1 * VdotH) / max(EPSILON, NdotV);

    // --- Calculate final PDF ---
    if (reflect)
    {
        // pdf_reflect = pdf_H / jacobian_reflect
        float pdf_reflect = pdf_H / (4.0f * VdotH);
        return prob_reflect * pdf_reflect;
    }
    else // Transmit
    {
        if (metalness > 0.0f) return 0.0f; // Metals don't transmit

        float LdotH = dot(L, H);
        // Transmitted L must be in lower hemisphere of H
        if (LdotH >= 0.0f) return 0.0f;

        // Refraction Jacobian: dH/dwi
        float denom = (etai * VdotH + etat * LdotH);
        float jacobian = (etat * etat * abs(LdotH)) / (denom * denom);

        if (jacobian == 0.0f) return 0.0f;

        // pdf_transmit = pdf_H / jacobian_transmit
        float pdf_transmit = pdf_H / jacobian;
        return prob_transmit * pdf_transmit;
    }
}

