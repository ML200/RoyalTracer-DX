inline float ESS_LUT(MaterialOptimized mat, float NdotV)
{
    // Normalize inputs to [0, 1]
    NdotV = saturate(NdotV);

    // Compute fractional index for the angle (NdotV)
    float thetaIdxF = NdotV * (LUT_SIZE_THETA - 1);

    // Compute integer indices for interpolation
    int thetaIdx0 = (int)floor(thetaIdxF);
    int thetaIdx1 = min(thetaIdx0 + 1, LUT_SIZE_THETA - 1);

    // Compute interpolation weight
    float wTheta = thetaIdxF - thetaIdx0;

    // Fetch LUT values at the two angle indices
    float v0 = materials[mat.mID].LUT[thetaIdx0];
    float v1 = materials[mat.mID].LUT[thetaIdx1];

    // Perform linear interpolation
    return lerp(v0, v1, wTheta);
    //return v0;
}


inline float3 SchlickFresnel(float3 F0, float cosTheta)
{
    return saturate(F0 + (1.0f - F0) * pow(abs(1.0f - cosTheta), 5.0f));
}

inline float D_GGX(float NdotH, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (alpha2 - 1.0f) + 1.0f);
    denom = max(denom, 1e-7f);
    return alpha2 / (PI * denom * denom);
}

// Smith's Geometry function for GGX
inline float G2_SmithGGX(float NdotV, float NdotL, float alpha)
{
    // Calculate the Smith masking term for the view direction
    float alpha2 = alpha * alpha;

    float denomA = NdotV * sqrt(alpha2 + (1.0f - alpha2) * NdotL * NdotL);
    float denomB = NdotL * sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV);

    return 2.0f * NdotL * NdotV / (denomA + denomB);
}

// Smith's Geometry function 2 for GGX
inline float G1_SmithGGX(float NdotV, float alpha)
{
      float alpha2 = alpha * alpha;
      float denomC = sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV) + NdotV;

      return 2.0f * NdotV / denomC;
}


// Coordinate system for transforming vectors
inline void CoordinateSystem(float3 N, out float3 T, out float3 B)
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
// Sample the BRDF of the given material using Heitz's VNDF sampling
inline void SampleBRDF_GGX(MaterialOptimized mat, float3 outgoing, float3 normal, float3 flatNormal, inout float3 sample, inout float3 origin, float3 worldOrigin, inout uint2 seed)
{
    float alpha = mat.Pr_Pm_Ps_Pc.x * mat.Pr_Pm_Ps_Pc.x;
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);
    float e0 = RandomFloat(seed);
    float e1 = RandomFloat(seed);

    float3 T1, T2;
    CoordinateSystem(N, T1, T2);
    float3 Vh = normalize(float3(dot(T1, V), dot(T2, V), dot(N, V)));
    if (Vh.z < 0.0f)
    {
        Vh = -Vh;
    }

    float alpha_x = alpha;
    float alpha_y = alpha;
    float3 Vh_stretched = normalize(float3(alpha_x * Vh.x, alpha_y * Vh.y, Vh.z));

    // Orthonormal basis
    float lensq = Vh_stretched.x * Vh_stretched.x + Vh_stretched.y * Vh_stretched.y;
    float3 T1h, T2h;
    if (lensq > 0.0f)
    {
        T1h = float3(-Vh_stretched.y, Vh_stretched.x, 0.0f) / sqrt(lensq);
        T2h = cross(Vh_stretched, T1h);
    }
    else
    {
        T1h = float3(1.0f, 0.0f, 0.0f);
        T2h = float3(0.0f, 1.0f, 0.0f);
    }

    // Sample point on disk
    float r = sqrt(e0);
    float phi = 2.0f * PI * e1;
    float x = r * cos(phi);
    float y = r * sin(phi);

    // Compute normal in stretched hemisphere
    float3 Nh_stretched = x * T1h + y * T2h + sqrt(max(0.0f, 1.0f - x * x - y * y)) * Vh_stretched;
    float3 Nh = normalize(float3(alpha_x * Nh_stretched.x, alpha_y * Nh_stretched.y, Nh_stretched.z));
    float3 H = Nh.x * T1 + Nh.y * T2 + Nh.z * N;
    sample = reflect(-V, H);

    // Check if the sample is in the same hemisphere as the surface normal
    if (dot(sample, N) <= 0.0f)
    {
        sample = float3(0.0f, 0.0f, 0.0f); // Invalid sample
    }
    origin = worldOrigin + s_bias * flatNormal;
}

// Evaluate the GGX BRDF for the given material
inline float3 EvaluateBRDF_GGX(MaterialOptimized mat, float3 normal, float3 incoming, float3 outgoing)
{
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);   // View direction
    float3 L = normalize(-incoming);  // Light direction
    float3 H = normalize(V + L);
    float NdotV = max(dot(N, V), EPSILON);
    float NdotL = max(dot(N, L), EPSILON);
    float NdotH = max(dot(N, H), EPSILON);
    float VdotH = max(dot(V, H), EPSILON);

    float3 F = SchlickFresnel(mat.Ks, VdotH);
    float D = D_GGX(NdotH, mat.Pr_Pm_Ps_Pc.x);
    float G = G2_SmithGGX(NdotV, NdotL, mat.Pr_Pm_Ps_Pc.x * mat.Pr_Pm_Ps_Pc.x);

    // Specular BRDF
    float denominator = 4.0f * NdotV * NdotL;
    if(denominator < EPSILON)
        return float3(0,0,0);

    float3 specular = (F * D * G) / denominator;

    //Multiscatter GGX
    float Ess = ESS_LUT(mat, NdotV);
    float kms = (1.0f - Ess) / Ess;

    float3 specular_ess = specular * (1.0f + mat.Ks * kms);
    return specular_ess;
}

// Calculate the PDF for a given sample direction using GGX
inline float BRDF_PDF_GGX(MaterialOptimized mat, float3 normal, float3 incoming, float3 outgoing)
{
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);   // View direction
    float3 L = normalize(-incoming);  // Light direction
    float3 H = normalize(V + L);
    float NdotH = max(dot(N, H), EPSILON);
    float NdotV = max(dot(N, V), EPSILON);

    float alpha = mat.Pr_Pm_Ps_Pc.x * mat.Pr_Pm_Ps_Pc.x;
    float G1 = G1_SmithGGX(NdotV, alpha);
    float D = D_GGX(NdotH, mat.Pr_Pm_Ps_Pc.x);

    return G1 * D / (NdotV * 4.0f);
}
