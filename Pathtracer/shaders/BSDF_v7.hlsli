// Select a sampling strategy for the given material:
// 0 - Lambertian
// 1 - Specular (GGX)
// 2 - Perfect reflection
// 3 - Refraction
// Probability is the likelihood to select the given sampling strategy, used for weighting the contributions
inline uint SelectSamplingStrategy(uint mID, float3 outgoing, float3 normal, inout uint2 seed, inout float probability){
    //Get random value
    float r = RandomFloat(seed);

    // Evaluate the material properties
    float roughness = materials[mID].Pr_Pm_Ps_Pc.x;
    float metallic = materials[mID].Pr_Pm_Ps_Pc.y;
    float clearcoat = materials[mID].Pr_Pm_Ps_Pc.w;
    float alpha = materials[mID].Kd.w;


    // Check if the ray will enter the material
    // First the Fresnel term has to be evaluated: Get the wavelength specific one and calculate the average
    //cos
    float cosTheta = dot(normal, outgoing);
    float3 fresnel = SchlickFresnel(materials[mID].Ks, cosTheta);

    // Sampling probabilities
    float p_s = min(1.0f, (fresnel.x + fresnel.y + fresnel.z)/3.0f + metallic); // Sample the specular part: grazing angles/ clearcoat for additive reflection (roughness) / metallic (will introduce colored reflections)
    float p_d = (1.0f - p_s); // Sample the diffuse part of the lobe

    //Adjust for translucency
    //p_d *= alpha;
    probability = p_s;

    //Select the strategy based on the probabilities (CDF)
    //Specular
    if(r <= p_s){
        if(roughness < 0.04f){ // adjust threshold (later 2)
            return 0;
        }
        return 1;
    }
    //Diffuse
    else if(r <= p_s + p_d){
        return 0;
    }
    else{
        // Refraction, currently replaced by diffuse (later 3)
        return 0;
    }
}

inline float2 CalculateStrategyProbabilities(uint mID, float3 outgoing, float3 normal){
    // Evaluate the material properties
    float roughness = materials[mID].Pr_Pm_Ps_Pc.x;
    float metallic  = materials[mID].Pr_Pm_Ps_Pc.y;
    // Note: clearcoat and alpha are not used in this calculation

    // Evaluate Fresnel term using Schlick's approximation
    float cosTheta = dot(normal, outgoing);
    float3 fresnel = SchlickFresnel(materials[mID].Ks, cosTheta);

    // Calculate the specular probability (strategy 1)
    float p_s = min(1.0f, (fresnel.x + fresnel.y + fresnel.z) / 3.0f + metallic);

    // Calculate the diffuse probability (strategy 0)
    float p_d = 1.0f - p_s;

    // Return the probabilities:
    // - x component: diffuse (strategy 0)
    // - y component: specular (strategy 1)
    return float2(p_d, p_s);
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
        //SampleBTDF_GGX(mat, incoming, normal, flatNormal, sample, origin, worldOrigin, seed);
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
        //return EvaluateBTDF_GGX(mID, normal, incidence, outgoing);
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
        //return BTDF_PDF_GGX(mat, normal, incidence, outgoing);
    }
    return 0;
}
