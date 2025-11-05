// Evaluate the GGX BRDF for the given material
inline float3 EvaluateBRDF_GGX_Dielectric(uint mID, float3 normal, float3 incoming, float3 outgoing, float etai, float etat)
{
    if(dot(normal, -incoming) < 0.0f)
        return (float3).0f;
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);
    float3 L = normalize(-incoming);

    float NdotV = max(0.0f, dot(N, V));
    float NdotL = max(0.0f, dot(N, L));
    if (NdotV <= 0.0f || NdotL <= 0.0f) return 0.0.xxx;

    float3 H = normalize(V + L);
    float NdotH = max(0.0f, dot(N, H));
    float VdotH = max(0.0f, dot(V, H));

    // Fresnel
    float3 F0_d = ComputeF0Dielectric(etai, etat);
    float3 F0_c = materials[mID].Kd,xyz;
    float3 F_d  = FresnelDielectricTIR(V, H, etai, etat); // dielectric fresnel
    float3 F_c  = FresnelConductor(F0_c, V, H); // conductor fresnel

    float alpha = materials[mID].Pr_Pm_Ps_Pc.x;
    alpha *= alpha; // alpha = roughness^2

    float D = D_GGX(NdotH, alpha);
    float G = G2_SmithGGX(NdotV, NdotL, alpha);

    float denom = 4.0f * NdotV * NdotL;
    float3 specular_d = (F_d * D * G) / denom; // dielectric
    float3 specular_c = (F_c * D * G) / denom; // conductor

    // Multiscatter compensation dielectric
    float Ess  = ESS_LUT(mID, NdotV);
    float kms  = (1.0f - Ess) / Ess;
    float3 specular_ess = (1.0f-materials[mID].Pr_Pm_Ps_Pc.y) * specular_d * (1.0f + F0_d * kms)
                           + materials[mID].Pr_Pm_Ps_Pc.y * specular_c * (1.0f + F0_c * kms);

    return (any(isnan(specular_ess)) || any(isinf(specular_ess))) ? 0.0.xxx : specular_ess;
}

// etaR is the ior of the material the ray is currently in before hitting the material
inline float Transmittance_GGX_Dielectric(uint mID, float3 normal, float3 incoming, float3 outgoing, float etai, float etat){
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

    // Metals dont transmit any energy
    return (1.0f - materials[mID].Pr_Pm_Ps_Pc.y) * (1.0f - F_o) * (1.0f - F_i) * (1.0f / (1.0f - Kd_frac * Favg));
}

// Sampling weight function describes an approximation of the transmittance before sampling, as incoming isnt known yet
inline float Sampling_Weight_GGX_Dielectric(uint mID, float3 normal, float3 outgoing, float etai, float etat){
    // We approximate here with a fresnel term coupled with the metalness of the material
    return (1.0f-materials[mID].Pr_Pm_Ps_Pc.y) * FresnelDielectric(outgoing, normal, etat, etai).x + materials[mID].Pr_Pm_Ps_Pc.y * Luma(FresnelConductor(materials[mID].Kd.xyz,outgoing, normal));
}


// VNDF sampling GGX
inline float3 SampleBRDF_GGX(
    uint mID,
    float3  outgoing,
    float3  normal,
    float3  flatNormal,
    inout uint2 seed)
{
    float r = materials[mID].Pr_Pm_Ps_Pc.x;
    float alpha = r * r;

    float3 V = normalize(outgoing);
    float3 N = normalize(normal);

    // 1. Sample the microfacet normal
    float3 H = SampleVNDF_H(alpha, V, N, seed);

    // 2. Reflect across it
    float3 sample = reflect(-V, H);

    if(dot(sample, N) <= 0.0f)
        sample = float3(0,0,0);
    return sample;
}

// Calculate the PDF for a given sample direction using GGX
inline float BRDF_PDF_GGX(uint mID, float3 N, float3 wi, float3 wo)
{
    if(dot(N, -wi) < 0.0f)
        return .0f;
    float3 V = normalize(wo);
    float3 L = normalize(-wi);
    float  NdotV = max(0.0f, dot(N, V));
    float  NdotL = max(0.0f, dot(N, L));
    if (NdotV <= 0.0f || NdotL <= 0.0f) return 0.0f;

    float3 H = normalize(V + L);

    float  r = materials[mID].Pr_Pm_Ps_Pc.x;
    float  alpha = max(EPSILON, r*r);
    float  D  = D_GGX(max(0.0f, dot(N, H)), alpha);
    float  G1 = G1_SmithGGX(NdotV, alpha);

    return (D * G1) / max(4.0f * NdotV, EPSILON);
}

