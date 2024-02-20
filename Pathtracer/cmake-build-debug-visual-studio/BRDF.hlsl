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
    float3 H = normalize(V + L); // Halfway vector
    float NoV = max(dot(N, V), 0.0f); // Normal and View vector dot product
    float NoL = max(dot(N, L), 0.0f); // Normal and Light vector dot product
    float NoH = max(dot(N, H), 0.0f); // Normal and Halfway vector dot product
    float VoH = max(dot(V, H), 0.0f); // View and Halfway vector dot product

    // GGX Distribution
    float D = GGXDistribution(alpha, NoH);

    // Geometry function
    float G = GeometrySmith(NoV, NoL, alpha);

    // Fresnel function using Schlick's approximation
    float3 F = FresnelSchlick(VoH, F0);

    // Specular BRDF
    float3 specular = (D * G * F) / (4.0f * NoV * NoL);

    return specular;
}


//Decide how to evaluate the material: distinguish between metals and dielectricts for simplicity
//This function returns the surface color evaluated
float3 evaluateBRDF(Material mat, float3 incidence, float3 normal, float3 light, inout float3 sample, inout float pdf, inout uint seed){
    //Check for metallicness
    if(mat.Pr_Pm_Ps_Pc.y > 0.5f) //Normally only 0 or 1
    {
        //Get the BRDF based on the materials properties
        sample =SampleGGXVNDF(normal,-incidence, mat.Pr_Pm_Ps_Pc.x,seed, pdf);
        return BRDF_Specular_GGX(normal, -incidence, light, CalculateF0Vector(1.45f),mat.Pr_Pm_Ps_Pc.x*mat.Pr_Pm_Ps_Pc.x);
    }
    else{
        //Calculate the fresnel term based on mat.Ni, incidence and normal
        //float f = SchlickFresnel(mat.Ni, incidence, normal); //For now hardcoded
        float f = SchlickFresnel(1.0f, -incidence, normal);
        //Get a random number
        float randomCheck = RandomFloat(seed);
        //Ckeck if the ray is diffuse reflected or through ggx:
        if(randomCheck < f){
            //Get the BRDF based on the materials properties
            sample =SampleGGXVNDF(normal, -incidence, mat.Pr_Pm_Ps_Pc.x,seed, pdf);
            return BRDF_Specular_GGX(normal, -incidence, light, CalculateF0Vector(1.0f),mat.Pr_Pm_Ps_Pc.x*mat.Pr_Pm_Ps_Pc.x);
        }
        else{
            //Diffuse reflection using lambertian
            sample = RandomUnitVectorInHemisphere(normal,seed); //already implemented
            pdf = 1.0f;
            return float3(1,1,1);
        }
    }
}