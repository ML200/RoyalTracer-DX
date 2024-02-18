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



//Decide how to evaluate the material: distinguish between metals and dielectricts for simplicity
//This function returns the surface color evaluated
float3 evaluateBRDF(Material mat, float3 incidence, float3 normal, float3 light, inout float3 sample, inout float pdf, inout uint seed){
    //Check for metallicness
    if(mat.Pr_Pm_Ps_Pc.y > 0.5f) //Normally only 0 or 1
    {
        //Get the BRDF based on the materials properties
        sample =SampleCone(normal,Reflect(incidence,normal), mat.Pr_Pm_Ps_Pc.x,seed, pdf);
        return mat.Kd;
    }
    else{
        //Calculate the fresnel term based on mat.Ni, incidence and normal
        //float f = SchlickFresnel(mat.Ni, incidence, normal); //For now hardcoded
        float f = SchlickFresnel(1.45f, -incidence, normal);
        //Get a random number
        float randomCheck = RandomFloat(seed);
        //Ckeck if the ray is diffuse reflected or through ggx:
        if(randomCheck < f){
            //Get the BRDF based on the materials properties
            sample =SampleCone(normal,Reflect(incidence,normal), mat.Pr_Pm_Ps_Pc.x,seed, pdf);
            return 1.0f;
        }
        else{
            //Diffuse reflection using lambertian
            sample = RandomUnitVectorInHemisphere(normal,seed); //already implemented
            pdf = 1.0f;
            return mat.Kd;
        }
    }
}