#include "Common.hlsl"

// Schlick Fresnel approximation
float3 SchlickFresnel(float3 F0, float cosTheta)
{
    return F0 + (1.0f - F0) * pow(abs(1.0f - cosTheta), 5.0f);
}

// GGX normal distribution function
float D_GGX(float NdotH, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (alpha2 - 1.0f) + 1.0f);
    return alpha2 / (PI * denom * denom);
}

//Geom schlick
float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = (roughness + 1.0);
    float k = (r*r) / 8.0;

    float nom   = NdotV;
    float denom = NdotV * (1.0 - k) + k;

    return nom / denom;
}

// Smith's Geometry function for GGX
float G_SmithGGX(float NdotV, float NdotL, float alpha) {
    float alpha2 = alpha * alpha;

    // GGX geometry term for view direction
    float GGXV = NdotV / (NdotV * (1.0 - alpha2) + alpha2);

    // GGX geometry term for light direction
    float GGXL = NdotL / (NdotL * (1.0 - alpha2) + alpha2);

    // Combine the terms
    return GGXV * GGXL;
}

// Coordinate system for transforming vectors
void CoordinateSystem(float3 N, out float3 T, out float3 B)
{
    if (abs(N.z) < 0.999f)
    {
        T = normalize(cross(float3(0.0f, 0.0f, 1.0f), N));
    }
    else
    {
        T = normalize(cross(float3(1.0f, 0.0f, 0.0f), N));
    }
    B = cross(N, T);
}

//______________________________________________________________________________________________________________________
// Sample the BRDF of the given material using GGX importance sampling
void SampleBRDF_GGX(Material mat, float3 outgoing, float3 normal, float3 flatNormal, inout float3 sample, inout float3 origin, float3 worldOrigin, inout uint2 seed)
{
    // Initialize variables
    float alpha = mat.Pr_Pm_Ps_Pc.x; // Roughness parameter
    float alpha2 = alpha * alpha;
    float3 N = normalize(normal); // Surface normal
    float3 T, B;
    CoordinateSystem(N, T, B); // Build tangent and bitangent vectors

    uint counter = 0;

    // Loop until a valid sample is generated
    while (true)
    {
        counter++;
        // Generate random numbers
        float e0 = RandomFloat(seed);
        float e1 = RandomFloat(seed);

        // Compute phi and theta for GGX sampling
        float phi = 2.0f * PI * e0;
        float cosTheta = sqrt((1.0f - e1) / (1.0f + (alpha2 - 1.0f) * e1));
        float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

        // Microfacet normal H in tangent space
        float3 H_tangent = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

        // Transform H_tangent to world space
        float3 H = H_tangent.x * T + H_tangent.y * B + H_tangent.z * N;

        // Reflect the incoming vector about H to get the outgoing vector
        sample = reflect(-outgoing, H);

        // Check if the sample is in the same hemisphere as the surface normal
        if (dot(sample, N) > 0.0f)
        {
            break; // Valid sample
        }
        if(counter > 5){
            break;
        }
    }

    // Offset the origin slightly along the flat normal to prevent self-intersection
    origin = worldOrigin + s_bias * flatNormal;
}


// Evaluate the GGX BRDF for the given material
float3 EvaluateBRDF_GGX(Material mat, float3 normal, float3 incoming, float3 outgoing)
{
    // Ensure the vectors are normalized
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);   // View direction
    float3 L = normalize(-incoming);  // Light direction
    float3 H = normalize(V + L);

    // Dot products
    float NdotV = abs(dot(N, V));
    float NdotL = abs(dot(N, L));
    float NdotH = abs(dot(N, H));
    float VdotH = abs(dot(V, H));

    // Fresnel term using Schlick's approximation
    float3 F0 = mat.Ks; // We interpolate between the dielectric specular and metallic specular
    float3 F = SchlickFresnel(F0, VdotH);

    // Normal Distribution Function (NDF)
    float D = D_GGX(NdotH, mat.Pr_Pm_Ps_Pc.x);
    //D = max(D, 1e-2f);

    // Geometry function
    float G = G_SmithGGX(NdotV, NdotL, mat.Pr_Pm_Ps_Pc.x);

    // Specular BRDF
    float denominator = 4.0f * NdotV * NdotL + 1e-7f;
    float3 specular = (F * D * G) / denominator;


    return specular;
}

// Calculate the PDF for a given sample direction using GGX
float BRDF_PDF_GGX(Material mat, float3 normal, float3 incoming, float3 outgoing)
{
    // Ensure the vectors are normalized
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);   // View direction
    float3 L = normalize(-incoming);  // Light direction
    float3 H = normalize(V + L);

    // Dot products
    float NdotH = abs(dot(N, H));
    float VdotH = abs(dot(V, H));

    // Normal Distribution Function (NDF)
    float D = D_GGX(NdotH, mat.Pr_Pm_Ps_Pc.x);

    // PDF calculation
    float pdf = (D * NdotH) / (4.0f * VdotH + 1e-7f);

    return pdf;
}
//______________________________________________________________________________________________________________________