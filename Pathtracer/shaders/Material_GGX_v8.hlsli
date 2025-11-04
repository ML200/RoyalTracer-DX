inline float ESS_LUT(uint mID, float NdotV)
{
    NdotV = saturate(NdotV);
    float thetaIdxF = NdotV * (LUT_SIZE - 1);

    int thetaIdx0 = (int)floor(thetaIdxF);
    int thetaIdx1 = min(thetaIdx0 + 1, LUT_SIZE - 1);

    // Interpolate between lut entries
    float wTheta = thetaIdxF - thetaIdx0;

    float v0 = materials[mID].LUT[thetaIdx0];
    float v1 = materials[mID].LUT[thetaIdx1];
    return lerp(v0, v1, wTheta);
}

// Evaluate the GGX BRDF for the given material
inline float3 EvaluateBRDF_GGX(uint mID, float3 normal, float3 incoming, float3 outgoing, float etai, float etat)
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
    float3 F0 = ComputeF0(mID, etai, etat);
    float3 F  = FresnelSchlickTIR(F0, V, H, etai, etat);

    float alpha = materials[mID].Pr_Pm_Ps_Pc.x;
    alpha *= alpha; // alpha = roughness^2

    float D = D_GGX(NdotH, alpha);
    float G = G2_SmithGGX(NdotV, NdotL, alpha);

    float denom = max(4.0f * NdotV * NdotL, EPSILON);
    float3 specular = (F * D * G) / denom;

    // Multiscatter compensation
    float Ess  = ESS_LUT(mID, NdotV);
    float kms  = (1.0f - Ess) / max(Ess, EPSILON);
    float3 specular_ess = specular * ((float3)1.0f + F0 * kms);

    // dielectric F0 (scalar) -> hemisphere average
    /*float  F0_diel = IORtoF0(materials[mID].Ni);
    float  Favg    = F0_diel + (1.0f - F0_diel) * (1.0f/21.0f);
    transmittance = (1.0f - materials[mID].Pr_Pm_Ps_Pc.y) * (1.0f - Favg);*/

    return (any(isnan(specular_ess)) || any(isinf(specular_ess))) ? 0.0.xxx : specular_ess;
}

// etaR is the ior of the material the ray is currently in before hitting the material
inline float Transmittance_GGX(uint mID, float3 normal, float3 incoming, float3 outgoing, float etai, float etat){
    // Transmittance is (1-Fo)(1-Fi)(1/(1-Kd*Favg)(1-metallicness)
    float3 F0 = ComputeF0(mID, etai, etat);
    float  Favg    = F0 + (1.0f - F0) * (1.0f/21.0f);

    return 1.0f;
}

// Sampling weight function describes an approximation of the transmittance before sampling, as incoming isnt known yet
inline float Sampling_Weight_GGX(uint mID, float3 normal, float3 outgoing, float etai, float etat){
    // We approximate here with a fresnel term coupled with the metalness of the material
    return 1.0f;
}


// VNDF sampling GGX
inline float3 SampleBRDF_GGX(uint mID, float3  outgoing, float3  normal, float3  flatNormal, inout uint2 seed)
{
    return SampleVNDF_GGX(mID, outgoing, normal, flatNormal, seed);
}

// Calculate the PDF for a given sample direction using GGX
inline float BRDF_PDF_GGX(uint mID, float3 N, float3 wi, float3 wo)
{
    if(dot(N, -wi) < 0.0f)
        return (float3).0f;
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

