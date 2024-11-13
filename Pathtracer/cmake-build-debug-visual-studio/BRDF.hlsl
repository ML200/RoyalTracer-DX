#include "Lambertian.hlsl"


// Select a sampling strategy for the given material:
// 0 - Lambertian
// 1 - Specular (GGX)
// 2 - Perfect reflection
// 3 - Refraction
// Probability is the likelihood to select the given sampling strategy, used for weighting the contributions
uint SelectSamplingStrategy(Material mat, float3 incoming, float3 normal, inout float probability, inout uint2 seed){
    //Get random value
    float r_1 = RandomFloat(seed);

    // Evaluate the material properties
    float roughness = mat.Pr_Pm_Ps_Pc.x;
    float metallic = mat.Pr_Pm_Ps_Pc.y;
    float sheen = mat.Pr_Pm_Ps_Pc.z;
    float clearcoat = mat.Pr_Pm_Ps_Pc.w;

    float alpha = mat.Kd.w;
    float3 specular = mat.Ks;

    // Check if the ray will enter the material
    // First the Fresnel term has to be evaluated: Get the wavelength specific one and calculate the average
    float3 fresnel_3 = SchlickFresnel(specular, dot(normal, -incident));
    float fresnel = (fresnel_3.x + fresnel_3.y + fresnel_3.z) / 3.0;




}

// Sample the BRDF of the given material
void SampleBRDF(uint strategy, Material mat, float3 incoming, float3 normal, float3 flatNormal, inout float3 sample, inout float3 origin, float3 worldOrigin, inout uint2 seed) {

}

// Evaluate the BRDF for the given material
float3 EvaluateBRDF(uint strategy, Material mat, float3 normal, float3 incidence, float3 outgoing) {

}

// Calculate the PDF for a given sample direction
float BRDF_PDF(uint strategy, Material mat, float3 normal, float3 incidence, float3 outgoing) {

}

// Combine the evaluation
float3 EvaluateBRDF_Combined(Material mat, float3 normal, float3 incidence, float3 outgoing) {

}

// Calculate the PDF for a given sample direction regarding the combination of pdfs
float BRDF_PDF_Combined(Material mat, float3 normal, float3 incidence, float3 outgoing) {

}