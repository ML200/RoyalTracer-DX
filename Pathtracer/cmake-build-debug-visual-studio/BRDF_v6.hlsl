// Select a sampling strategy for the given material:
// 0 - Lambertian
// 1 - Specular (GGX)
// 2 - Perfect reflection
// 3 - Refraction
// Probability is the likelihood to select the given sampling strategy, used for weighting the contributions
uint SelectSamplingStrategy(Material mat, float3 outgoing, float3 normal, inout uint2 seed, inout float probability){
    //Get random value
    float r = RandomFloat(seed);

    // Evaluate the material properties
    float roughness = mat.Pr_Pm_Ps_Pc.x;
    float metallic = mat.Pr_Pm_Ps_Pc.y;
    float clearcoat = mat.Pr_Pm_Ps_Pc.w;
    float alpha = mat.Kd.w;


    // Check if the ray will enter the material
    // First the Fresnel term has to be evaluated: Get the wavelength specific one and calculate the average
    //cos
    float cosTheta = dot(normal, outgoing);
    float3 fresnel = SchlickFresnel(mat.Ks, cosTheta);

    // Sampling probabilities
    float p_s = min(1.0f, (fresnel.x + fresnel.y + fresnel.z)/3.0f + clearcoat + metallic); // Sample the specular part: grazing angles/ clearcoat for additive reflection (roughness) / metallic (will introduce colored reflections)
    float p_d = (1.0f - p_s); // Sample the diffuse part of the lobe
    //Adjust for translucency
    p_d *= alpha;

    //Select the strategy based on the probabilities (CDF)
    //Specular
    if(r <= p_s){
        if(roughness < 0.04f){ // adjust threshold (later 2)
            return 0;
        }
        probability = p_s;
        return 1;
    }
    //Diffuse
    else if(r <= p_s + p_d){
        probability = p_d;
        return 0;
    }
    else{
        // Refraction, currently replaced by diffuse (later 3)
        probability = 1.0f - (p_d + p_s);
        return 0;
    }

}

// Sample the BRDF of the given strategy
void SampleBRDF(uint strategy, Material mat, float3 incoming, float3 normal, float3 flatNormal, inout float3 sample, inout float3 origin, float3 worldOrigin, inout uint2 seed) {
    //Sample from the selected strategy
    if(strategy == 0){
        SampleBRDF_Lambertian(mat, incoming, normal, flatNormal, sample, origin, worldOrigin, seed);
    }
    else if(strategy == 1){
        SampleBRDF_GGX(mat, incoming, normal, flatNormal, sample, origin, worldOrigin, seed);
    }
    else if(strategy == 2){

    }
    else{
        //SampleBTDF_GGX(mat, incoming, normal, flatNormal, sample, origin, worldOrigin, seed);
    }
}

// Evaluate the BRDF for the given strategy
float3 EvaluateBRDF(uint strategy, Material mat, float3 normal, float3 incidence, float3 outgoing) {
    //Sample from the selected strategy
    if(strategy == 0){
        return EvaluateBRDF_Lambertian(mat, normal, incidence, outgoing);
    }
    else if(strategy == 1){
        return EvaluateBRDF_GGX(mat, normal, incidence, outgoing);
    }
    else if(strategy == 2){

    }
    else{
        //return EvaluateBTDF_GGX(mat, normal, incidence, outgoing);
    }
    return (0,0,0);
}

// Calculate the PDF for a given sample direction and strategy
float BRDF_PDF(uint strategy, Material mat, float3 normal, float3 incidence, float3 outgoing) {
    //Sample from the selected strategy
    if(strategy == 0){
        return BRDF_PDF_Lambertian(mat, normal, incidence, outgoing);
    }
    else if(strategy == 1){
        return BRDF_PDF_GGX(mat, normal, incidence, outgoing);
    }
    else if(strategy == 2){

    }
    else{
        //return BTDF_PDF_GGX(mat, normal, incidence, outgoing);
    }
    return 0;
}

float3 EvaluateBRDF_Combined(Material mat, float3 normal, float3 incoming, float3 outgoing) {
    float3 brdf_total = float3(0.0f, 0.0f, 0.0f);

    // Evaluate individual BRDF components
    float3 brdf_diffuse = EvaluateBRDF_Lambertian(mat, normal, incoming, outgoing);
    float3 brdf_specular = EvaluateBRDF_GGX(mat, normal, incoming, outgoing);

    // Evaluate material properties
    float roughness = mat.Pr_Pm_Ps_Pc.x;
    float metallic = mat.Pr_Pm_Ps_Pc.y;
    float clearcoat = mat.Pr_Pm_Ps_Pc.w;
    float alpha = mat.Kd.w;

    // Calculate Fresnel term
    float3 fresnel = SchlickFresnel(mat.Ks, abs(dot(normal, outgoing)));

    // Sampling probabilities
    float p_s = min(1.0f, length(fresnel)/3.0f + clearcoat + metallic);
    float p_d = 1.0f - p_s;

    // Adjust for translucency
    p_s *= alpha;
    p_d *= alpha;

    // Normalize probabilities
    float total_p = p_s + p_d;
    if (total_p > 0.0f) {
        p_s /= total_p;
        p_d /= total_p;
    }

    // Sum the BRDF components
    brdf_total += brdf_diffuse * p_d;
    brdf_total += brdf_specular * p_s;

    return brdf_total;
}

float BRDF_PDF_Combined(Material mat, float3 normal, float3 incoming, float3 outgoing) {
    // Compute the individual PDFs
    float pdf_diffuse = BRDF_PDF_Lambertian(mat, normal, incoming, outgoing);
    float pdf_specular = BRDF_PDF_GGX(mat, normal, incoming, outgoing);

    // Evaluate material properties
    float roughness = mat.Pr_Pm_Ps_Pc.x;
    float metallic = mat.Pr_Pm_Ps_Pc.y;
    float clearcoat = mat.Pr_Pm_Ps_Pc.w;
    float alpha = mat.Kd.w;

    // Calculate Fresnel term
    float3 fresnel = SchlickFresnel(mat.Ks, abs(dot(normal, outgoing)));

    // Sampling probabilities
    float p_s = min(1.0f, (fresnel.x + fresnel.y + fresnel.z)/3.0f + clearcoat + metallic);
    float p_d = 1.0f - p_s;

    // Adjust for translucency
    p_d *= alpha;

    // Normalize probabilities
    float total_p = p_s + p_d;
    if (total_p > 0.0f) {
        p_s /= total_p;
        p_d /= total_p;
    }

    // Compute the combined PDF
    float pdf_combined = p_s * pdf_specular + p_d * pdf_diffuse;

    return pdf_combined;
}
