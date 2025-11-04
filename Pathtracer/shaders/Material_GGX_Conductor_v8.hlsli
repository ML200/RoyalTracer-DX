// Evaluate the GGX BRDF for the given material conductor
inline float3 EvaluateBRDF_GGX_Conductor(uint mID, float3 normal, float3 incoming, float3 outgoing, float etai, float etat)
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
    float3 F0 = ComputeF0Dielectric(etai, etat);
    float3 F  = FresnelDielectricTIR(V, H, etai, etat);

    float alpha = materials[mID].Pr_Pm_Ps_Pc.x;
    alpha *= alpha; // alpha = roughness^2

    float D = D_GGX(NdotH, alpha);
    float G = G2_SmithGGX(NdotV, NdotL, alpha);

    float denom = max(4.0f * NdotV * NdotL, EPSILON);
    float3 specular = (F * D * G) / denom;

    // Multiscatter compensation
    /*float Ess  = ESS_LUT(mID, NdotV);
    float kms  = (1.0f - Ess) / max(Ess, EPSILON);
    float3 specular_ess = specular * ((float3)1.0f + F0 * kms);*/

    return (any(isnan(specular)) || any(isinf(specular))) ? 0.0.xxx : specular;
}

// etaR is the ior of the material the ray is currently in before hitting the material
inline float Transmittance_GGX_Conductor(uint mID, float3 normal, float3 incoming, float3 outgoing, float etai, float etat){
    // Transmittance is (1-Fo)(1-Fi)(1/(1-Kd*Favg) -> only for dielectrics
    float F0 = ComputeF0Dielectric(etai, etat).x;
    float  Favg    = F0 + (1.0f - F0) * (1.0f/21.0f);

    // Compute the fresnel term for the incoming and outgoing ray
    float F_o = FresnelDielectric(outgoing, normal, etat, etai).x;
    float F_i = FresnelDielectric(-incoming, normal, etai, etat).x;

    // Material properties
    float Kd_frac = Avg3(materials[mID].Kd.xyz * materials[mID].Kd.w); // Fraction of the Kd layer of the material that might reflect back into the specular layer

    return (1.0f - F_o) * (1.0f - F_i) * (1.0f / (1.0f - Kd_frac * Favg));
}

// Sampling weight function describes an approximation of the transmittance before sampling, as incoming isnt known yet
inline float Sampling_Weight_GGX_Conductor(uint mID, float3 normal, float3 outgoing, float etai, float etat){
    // We approximate here with a fresnel term coupled with the metalness of the material
    return FresnelDielectric(outgoing, normal, etat, etai).x;
}


// VNDF sampling GGX
inline float3 SampleBRDF_GGX(uint mID, float3  outgoing, float3  normal, float3  flatNormal, inout uint2 seed)
{
    return SampleVNDF_GGX(mID, outgoing, normal, flatNormal, seed);
}
