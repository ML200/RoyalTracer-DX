// Tables store endpoint samples. Map them to texel centers before filtering.
inline float2 MaterialEnergyUV(float roughness, float NdotV)
{
    return (saturate(float2(roughness, NdotV)) * 15.0f + 0.5f) / 16.0f;
}

inline float GetSheenLUT(float roughness, float NdotV)
{
    float3 uvw = float3(MaterialEnergyUV(roughness, NdotV), SHEEN_LUT_INDEX);
    return g_LUT.SampleLevel(g_sampler_LUT, uvw, 0).r;
}

inline float GetEssLUT(float roughness, float NdotV)
{
    float3 uvw = float3(MaterialEnergyUV(roughness, NdotV), GGX_ESS_LUT_INDEX);
    return g_LUT.SampleLevel(g_sampler_LUT, uvw, 0).r;
}

// Integral of unit-Fresnel GGX times {1, (1-V.H)^5, (1-V.H)^10}.
// All three moments come from the same small texture fetch used for ESS.
inline float3 GetGGXEnergyMoments(float roughness, float NdotV)
{
    return g_LUT.SampleLevel(g_sampler_LUT,
        float3(MaterialEnergyUV(roughness, NdotV), GGX_ESS_LUT_INDEX), 0).rgb;
}

inline float GGXDirectionalReflectance(float roughness, float NdotV, float etai, float etat,
                                     bool coat = false)
{
    // The legacy TIR branch can only increase reflection: reserve its upper bound.
    if (etai > etat) return 1.0f;
    float3 e = GetGGXEnergyMoments(roughness, NdotV);
    float f0 = ComputeF0Dielectric(etai, etat).x;
    float f1 = 1.0f - f0;
    float reflected = f0 * e.x + f1 * e.y;
    float kms = (1.0f - e.x) / max(e.x, 1e-6f);
    if (coat)
        reflected += kms * (f0 * f0 * e.x + 2.0f * f0 * f1 * e.y + f1 * f1 * e.z);
    else
        reflected *= 1.0f + f0 * kms;
    return saturate(reflected);
}

inline float D_GGX(float3 N, float3 H, float alpha)
{
    const float alpha2 = alpha * alpha;
    const float NdotH = dot(N, H);
    const float3 NxH = cross(N, H);

    const float denom = dot(NxH, NxH) + alpha2 * NdotH * NdotH;
    return alpha2 / (PI * denom * denom);
}

// Burley 2012 / Heitz 2014.
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

// Heitz 2014.
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

inline void ComputeAnisotropicAlphas(float alpha, float aniso, out float ax, out float ay)
{
    float aspect = sqrt(1.0f - 0.9f * abs(aniso));
    ax = max(0.001f, alpha / aspect);
    ay = max(0.001f, alpha * aspect);
}

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

// Sample an anisotropic visible-normal distribution.
inline float3 SampleVNDF_H_Aniso(float alpha_x, float alpha_y, float3 V, float3 N, float3 T1, float3 T2, inout uint seed)
{

    float vx = dot(T1, V);
    float vy = dot(T2, V);
    float vz = abs(dot(N,  V)) + 0.00001f;

    float3 Ve = normalize(float3(alpha_x * vx, alpha_y * vy, vz));

    float lensq = Ve.x * Ve.x + Ve.y * Ve.y;
    float3 T1h = (lensq > 0.0f)
               ? float3(-Ve.y, Ve.x, 0.0f) * rsqrt(lensq)
               : float3(1.0f, 0.0f, 0.0f);
    float3 T2h = cross(Ve, T1h);

    float U1  = RandomFloatSingle(seed);
    float U2  = RandomFloatSingle(seed);
    float r   = sqrt(U1);
    float phi = 2.0f * PI * U2;
    float t1  = r * cos(phi);
    float t2  = r * sin(phi);
    float s   = 0.5f * (1.0f + Ve.z);
    t2 = (1.0f - s) * sqrt(saturate(1.0f - t1 * t1)) + s * t2;

    float3 Nh = t1 * T1h + t2 * T2h + sqrt(saturate(1.0f - t1 * t1 - t2 * t2)) * Ve;
    float3 Ne = float3(alpha_x * Nh.x, alpha_y * Nh.y, max(0.0f, Nh.z));

    return normalize(Ne.x * T1 + Ne.y * T2 + Ne.z * N);
}

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
