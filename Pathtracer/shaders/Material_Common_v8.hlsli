inline float GetSheenLUT(float roughness, float NdotV)
{
    float3 uvw = float3(roughness, saturate(NdotV), SHEEN_LUT_INDEX);
    return g_LUT.SampleLevel(g_sampler_LUT, uvw, 0).r;
}

inline float GetEssLUT(float roughness, float NdotV)
{
    float3 uvw = float3(roughness, saturate(NdotV), GGX_ESS_LUT_INDEX);
    return g_LUT.SampleLevel(g_sampler_LUT, uvw, 0).r;
}



inline float D_GGX(float NdotH, float alpha)
{
    float alpha2 = alpha * alpha;
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (alpha2 - 1.0f) + 1.0f);
    return alpha2 / (PI * denom * denom);
}

// Anisotropic GGX NDF (Burley 2012 / Heitz 2014)
inline float D_GGX_Aniso(float NdotH, float TdotH, float BdotH, float ax, float ay)
{
    float tx = TdotH / ax;
    float by = BdotH / ay;
    float d  = tx * tx + by * by + NdotH * NdotH;
    return 1.0f / (PI * ax * ay * d * d);
}

inline float G1_SmithGGX(float NdotV, float alpha)
{
    float alpha2 = alpha * alpha;
    float denomC = sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV) + NdotV;

    return 2.0f * NdotV / denomC;
}

// Anisotropic Smith G1 (Heitz 2014)
inline float G1_SmithGGX_Aniso(float NdotV, float TdotV, float BdotV, float ax, float ay)
{
    float a2 = (ax * TdotV) * (ax * TdotV) + (ay * BdotV) * (ay * BdotV) + NdotV * NdotV;
    return 2.0f * NdotV / (NdotV + sqrt(a2));
}

inline float G2_SmithGGX(float NdotV, float NdotL, float alpha)
{
    return G1_SmithGGX(NdotV, alpha) * G1_SmithGGX(NdotL, alpha);
}

inline float G2_SmithGGX_Aniso(float NdotV, float TdotV, float BdotV,
                                float NdotL, float TdotL, float BdotL,
                                float ax, float ay)
{
    return G1_SmithGGX_Aniso(NdotV, TdotV, BdotV, ax, ay)
         * G1_SmithGGX_Aniso(NdotL, TdotL, BdotL, ax, ay);
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

// Compute anisotropic alpha values from isotropic roughness + anisotropy parameter
// aniso in [-1, 1]: 0 = isotropic, positive = stretch along tangent
inline void ComputeAnisotropicAlphas(float alpha, float aniso, out float ax, out float ay)
{
    float aspect = sqrt(1.0f - 0.9f * abs(aniso));
    ax = max(0.001f, alpha / aspect);
    ay = max(0.001f, alpha * aspect);
}

// Build tangent frame from normal, then rotate by anisotropy rotation angle
inline void BuildAnisotropicFrame(float3 N, float anisoRotation, out float3 T, out float3 B)
{
    CoordinateSystem(N, T, B);
    if (anisoRotation > 0.001f)
    {
        float angle = anisoRotation * PI;
        float ca = cos(angle);
        float sa = sin(angle);
        float3 T0 = T;
        T =  ca * T0 + sa * B;
        B = -sa * T0 + ca * B;
    }
}


// Sample the microfacet normal using VNDF sampling (anisotropic)
// T1, T2: tangent frame (possibly rotated for anisotropy)
inline float3 SampleVNDF_H_Aniso(float alpha_x, float alpha_y, float3 V, float3 N, float3 T1, float3 T2, inout uint seed)
{
    // hemisphere config (stretch view)
    float vx = dot(T1, V);
    float vy = dot(T2, V);
    float vz = abs(dot(N,  V)) + 0.00001f;

    float3 Ve = normalize(float3(alpha_x * vx, alpha_y * vy, vz));

    // build orthonormal basis
    float lensq = Ve.x * Ve.x + Ve.y * Ve.y;
    float3 T1h = (lensq > 0.0f)
               ? float3(-Ve.y, Ve.x, 0.0f) * rsqrt(lensq)
               : float3(1.0f, 0.0f, 0.0f);
    float3 T2h = cross(Ve, T1h);

    // sample disk and warp
    float U1  = RandomFloatSingle(seed);
    float U2  = RandomFloatSingle(seed);
    float r   = sqrt(U1);
    float phi = 2.0f * PI * U2;
    float t1  = r * cos(phi);
    float t2  = r * sin(phi);
    float s   = 0.5f * (1.0f + Ve.z);
    t2 = (1.0f - s) * sqrt(saturate(1.0f - t1 * t1)) + s * t2;

    // reprojection step
    float3 Nh = t1 * T1h + t2 * T2h + sqrt(saturate(1.0f - t1 * t1 - t2 * t2)) * Ve;
    float3 Ne = float3(alpha_x * Nh.x, alpha_y * Nh.y, max(0.0f, Nh.z));

    // convert to world space and return H
    return normalize(Ne.x * T1 + Ne.y * T2 + Ne.z * N);
}

// Isotropic convenience wrapper (backward compat)
inline float3 SampleVNDF_H(float alpha, float3 V, float3 N, inout uint seed)
{
    float3 T1, T2;
    CoordinateSystem(N, T1, T2);
    return SampleVNDF_H_Aniso(alpha, alpha, V, N, T1, T2, seed);
}

inline uint FlatPrimID(uint instID, uint geomIdx, uint primIdx)
{
    return (geomIdx == 0) ? primIdx : (instanceProps[instID].opaqueTriCount + primIdx);
}