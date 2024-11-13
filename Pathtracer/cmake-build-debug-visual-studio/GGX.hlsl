#include "Common.hlsl"
const float TwoPI = 6.28318530718f;

// Schlick Fresnel approximation
float3 SchlickFresnel(float3 F0, float cosTheta)
{
    return F0 + (1.0f - F0) * pow(1.0f - cosTheta, 5.0f);
}

// GGX normal distribution function
float D_GGX(float NdotH, float alpha)
{
    float alpha2 = alpha * alpha;
    float NdotH2 = NdotH * NdotH;
    float denom = (NdotH2 * (alpha2 - 1.0f) + 1.0f);
    return alpha2 / (PI * denom * denom);
}

// Smith's Geometry function for GGX
float G_SmithGGX(float NdotV, float NdotL, float alpha)
{
    float alpha2 = alpha * alpha;

    float GGXV = NdotV + sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV);
    float GGXL = NdotL + sqrt(alpha2 + (1.0f - alpha2) * NdotL * NdotL);

    return NdotV * NdotL / (GGXV * GGXL);
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
void SampleBRDF_GGX(Material mat, float3 incoming, float3 normal, float3 flatNormal, inout float3 sample, inout float3 origin, float3 worldOrigin, inout uint2 seed)
{
    // Generate random numbers
    float e0 = RandomFloat(seed);
    float e1 = RandomFloat(seed);

    // Roughness parameter
    float alpha = mat.Pr_Pm_Ps_Pc.x;
    float alpha2 = alpha * alpha;

    // Compute phi and theta for GGX sampling
    float phi = TwoPI * e0;
    float cosTheta = sqrt((1.0f - e1) / (1.0f + (alpha2 - 1.0f) * e1));
    float sinTheta = sqrt(1.0f - cosTheta * cosTheta);

    // Microfacet normal H in tangent space
    float3 H_tangent = float3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    // Build an orthonormal basis around the normal
    float3 N = normalize(normal);
    float3 T, B;
    CoordinateSystem(N, T, B);

    // Transform H_tangent to world space
    float3 H = H_tangent.x * T + H_tangent.y * B + H_tangent.z * N;

    // Reflect the incoming vector about H to get the outgoing vector
    sample = reflect(-incoming, H);

    // Ensure the sample is in the same hemisphere as the normal
    if (dot(sample, N) <= 0.0f)
    {
        // If not, discard the sample (you might want to resample instead)
        sample = float3(0.0f, 0.0f, 0.0f);
    }

    // Offset the origin slightly along the normal to prevent self-intersection
    origin = worldOrigin + s_bias * flatNormal;
}

// Evaluate the GGX BRDF for the given material
float3 EvaluateBRDF_GGX(Material mat, float3 normal, float3 incidence, float3 outgoing)
{
    // Ensure the vectors are normalized
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);   // View direction
    float3 L = normalize(incidence);  // Light direction
    float3 H = normalize(V + L);

    // Dot products
    float NdotV = saturate(dot(N, V));
    float NdotL = saturate(dot(N, L));
    float NdotH = saturate(dot(N, H));
    float VdotH = saturate(dot(V, H));

    // Fresnel term using Schlick's approximation
    float3 F0 = mat.Ks;
    float3 F = SchlickFresnel(F0, VdotH);

    // Normal Distribution Function (NDF)
    float D = D_GGX(NdotH, mat.Pr_Pm_Ps_Pc.x);

    // Geometry function
    float G = G_SmithGGX(NdotV, NdotL, mat.Pr_Pm_Ps_Pc.x);

    // Specular BRDF
    float denominator = 4.0f * NdotV * NdotL + 1e-7f;
    float3 specular = (F * D * G) / denominator;

    return specular;
}

// Calculate the PDF for a given sample direction using GGX
float BRDF_PDF_GGX(Material mat, float3 normal, float3 incidence, float3 outgoing)
{
    // Ensure the vectors are normalized
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);   // View direction
    float3 L = normalize(incidence);  // Light direction
    float3 H = normalize(V + L);

    // Dot products
    float NdotH = saturate(dot(N, H));
    float VdotH = saturate(dot(V, H));

    // Normal Distribution Function (NDF)
    float D = D_GGX(NdotH, mat.Pr_Pm_Ps_Pc.x);

    // PDF calculation
    float pdf = (D * NdotH) / (4.0f * abs(VdotH) + 1e-7f);

    return pdf;
}
//______________________________________________________________________________________________________________________