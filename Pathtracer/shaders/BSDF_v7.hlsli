// Principled sampling probabilities
inline float2 CalculateStrategyProbabilities(uint mID, float3 outgoing, float3 normal)
{
    float  roughness = materials[mID].Pr_Pm_Ps_Pc.x;
    float  metallic  = materials[mID].Pr_Pm_Ps_Pc.y;

    float3 N = normalize(normal);
    float3 V = normalize(outgoing);
    float  NdotV = saturate(dot(N, V));

    float3 F0_rgb = ComputeBaseF0(mID);
    float  F0     = saturate(Avg3(F0_rgb));

    // Fresnel at this view angle (achromatic for weights)
    float  Fv     = saturate(Avg3(SchlickFresnel(F0_rgb, NdotV)));

    // Angle-aware energy split
    float w_diff = (1.0 - metallic) * (1.0 - Fv);
    float w_spec = lerp(Fv, 1.0, metallic);

    // Small variance tilt toward spec on very smooth surfaces
    float glossBias = 1.0 - saturate(roughness);
    w_spec *= lerp(1.0, 1.25, glossBias * 0.5);

    float sum = max(1e-6, w_diff + w_spec);
    return float2(w_diff / sum, w_spec / sum); // (p_diff, p_spec)
}

// Select a sampling strategy for the given material:
// 0 - Lambertian
// 1 - Specular (GGX)
// 2 - Perfect reflection
// 3 - Refraction
// Probability is the likelihood to select the given sampling strategy, used for weighting the contributions
inline uint SelectSamplingStrategy(uint mID, float3 outgoing, float3 normal, inout uint2 seed)
{
    float r = RandomFloatSingle(seed.x);

    // Get principled (diffuse, specular) probabilities
    float2 ps = CalculateStrategyProbabilities(mID, outgoing, normal);
    float p_diff = saturate(ps.x);
    float p_spec = saturate(ps.y);

    // Ensure they form a valid distribution
    float sum = max(1e-6f, p_diff + p_spec);
    p_diff /= sum;
    p_spec /= sum;

    // CDF selection
    if (r < p_spec)
    {
        return 1; // Specular GGX
    }

    return 0; // Lambertian
}




// Sample the BRDF of the given strategy
inline void SampleBRDF(uint strategy, uint mID, float3 incoming, float3 normal, float3 flatNormal, inout float3 sample, float3 worldOrigin, inout uint2 seed) {
    //Sample from the selected strategy
    if(strategy == 0){
        SampleBRDF_Lambertian(mID, incoming, normal, flatNormal, sample, worldOrigin, seed);
    }
    else if(strategy == 1){
        SampleBRDF_GGX(mID, incoming, normal, flatNormal, sample, worldOrigin, seed);
    }
    else if(strategy == 2){

    }
    else{
        SampleBRDF_Lambertian(mID, incoming, normal, flatNormal, sample, worldOrigin, seed);
    }
}

// Evaluate the BRDF for the given strategy
inline float3 EvaluateBRDF(uint strategy, uint mID, float3 normal, float3 incidence, float3 outgoing) {
    //Sample from the selected strategy
    if(strategy == 0){
        return EvaluateBRDF_Lambertian(mID, normal, incidence, outgoing);
    }
    else if(strategy == 1){
        return EvaluateBRDF_GGX(mID, normal, incidence, outgoing);
    }
    else if(strategy == 2){

    }
    else{
        return EvaluateBRDF_Lambertian(mID, normal, incidence, outgoing);
    }
    return float3(0,0,0);
}

// Calculate the PDF for a given sample direction and strategy
inline float BRDF_PDF(uint strategy, uint mID, float3 normal, float3 incidence, float3 outgoing) {
    //Sample from the selected strategy
    if(strategy == 0){
        return BRDF_PDF_Lambertian(mID, normal, incidence, outgoing);
    }
    else if(strategy == 1){
        return BRDF_PDF_GGX(mID, normal, incidence, outgoing);
    }
    else if(strategy == 2){

    }
    else{
        return BRDF_PDF_Lambertian(mID, normal, incidence, outgoing);
    }
    return 0;
}
