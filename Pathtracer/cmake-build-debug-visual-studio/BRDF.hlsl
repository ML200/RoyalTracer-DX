#include "Common.hlsl"

float SchlickFresnel(float Ni, float3 incidence, float3 normal)
{
    // Calculate reflectance at normal incidence, F0, using the refractive index
    float F0 = pow((Ni - 1) / (Ni + 1), 2);

    // Calculate the cosine of the angle between the incidence direction and the normal
    float cosTheta = dot(normalize(incidence), normalize(normal));

    // Apply Schlick's approximation
    float F = F0 + (1 - F0) * pow(1 - cosTheta, 5);

    return F;
}

float3 BRDF_Specular_GGX(float3 N, float3 V, float3 L, float3 F0, float alpha) {
    // Avoid zero-length vectors for V + L
    float3 V_plus_L = V + L;
    if (length(V_plus_L) < 1e-4f) {
        return float3(0.0f, 0.0f, 0.0f); // Safety check for zero-length vectors
    }
    float3 H = normalize(V_plus_L);

    // Additional safety checks for grazing angles
    float NoV = max(dot(N, V), 1e-4f);
    float NoL = max(dot(N, L), 1e-4f);
    float NoH = max(dot(N, H), 1e-4f);
    float VoH = max(dot(V, H), 1e-4f);

    // Use safe GGX Distribution, Geometry, and Fresnel functions
    float D = GGXDistribution(alpha, NoH);
    float G = GeometrySmith(NoV, NoL, alpha);
    float3 F = FresnelSchlick(VoH, F0);

    // Ensure denominator is not zero
    float denom = max(4.0f * NoV * NoL, 1e-4f);
    float3 specular = (D * G * F) / denom;

    // Check for NaNs again if necessary
    if (any(isnan(specular))) {
        return float3(0.0f, 0.0f, 0.0f);
    }

    return specular;
}


//Decide how to evaluate the material: distinguish between metals and dielectricts for simplicity
//This function returns the surface color evaluated
float3 evaluateBRDF(Material mat, float3 incidence, float3 normal, float3 flatNormal, float3 light, inout float3 sample, inout float pdf, inout uint2 seed){
    //Check for metallicness
    if(mat.Pr_Pm_Ps_Pc.y > 0.5f) //Normally only 0 or 1
    {
        //Get the BRDF based on the materials properties
        sample =SampleGGXVNDF(normal,flatNormal,-incidence, mat.Pr_Pm_Ps_Pc.x,seed, pdf);
        return BRDF_Specular_GGX(normal, -incidence, light, mat.Kd,mat.Pr_Pm_Ps_Pc.x*mat.Pr_Pm_Ps_Pc.x);
    }
    else{
        //Calculate the fresnel term based on mat.Ni, incidence and normal
        //float f = SchlickFresnel(mat.Ni, incidence, normal); //For now hardcoded
        float f = SchlickFresnel(1.45f, -incidence, flatNormal) * (1-mat.Pr_Pm_Ps_Pc.x);
        //Get a random number
        float randomCheck = RandomFloatLCG(seed.x);
        //Ckeck if the ray is diffuse reflected or through ggx:
        if(randomCheck < f){
            //Get the BRDF based on the materials properties
            sample =SampleGGXVNDF(normal, flatNormal, -incidence, mat.Pr_Pm_Ps_Pc.x,seed, pdf);
            return BRDF_Specular_GGX(normal, -incidence, light, CalculateF0Vector(1.0f),mat.Pr_Pm_Ps_Pc.x*mat.Pr_Pm_Ps_Pc.x);
        }
        else{
            //Diffuse reflection using lambertian
            sample = RandomUnitVectorInHemisphere(normal,seed);
            pdf = 1.0f;
            return float3(1,1,1);
        }
    }
}