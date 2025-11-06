inline float ESS_LUT(uint mID, float NdotV)
{
    NdotV = saturate(NdotV);
    float thetaIdxF = NdotV * (LUT_SIZE - 1);

    int thetaIdx0 = (int)floor(thetaIdxF);
    int thetaIdx1 = min(thetaIdx0 + 1, LUT_SIZE - 1);

    // Interpolate between lut entries
    float wTheta = thetaIdxF - thetaIdx0;

    float v0 = materials[mID].LUT[thetaIdx0];
    float v1 = materials[mID].LUT[thetaIdx1];
    return lerp(v0, v1, wTheta);
}

inline float D_GGX(float NdotH, float alpha)
{
    float alpha2 = alpha * alpha;
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (alpha2 - 1.0f) + 1.0f);
    denom = denom;
    return alpha2 / (PI * denom * denom);
}

inline float G1_SmithGGX(float NdotV, float alpha)
{
    float alpha2 = alpha * alpha;
    float denomC = sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV) + NdotV;

    return 2.0f * NdotV / denomC;
}

inline float G2_SmithGGX(float NdotV, float NdotL, float alpha)
{
    return G1_SmithGGX(NdotV, alpha) * G1_SmithGGX(NdotL, alpha);
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


// Sample the microfacet normal using vndf sampling
inline float3 SampleVNDF_H(float alpha, float3 V, float3 N, inout uint2 seed)
{
    // world to local transform
    float3 T1, T2;
    CoordinateSystem(N, T1, T2);

    // hemisphere config (stretch view)
    float alpha_x = alpha;
    float alpha_y = alpha;
    float vx = dot(T1, V);
    float vy = dot(T2, V);
    float vz = dot(N,  V);

    float3 Ve = normalize(float3(alpha_x * vx, alpha_y * vy, vz));

    // build orthonormal basis
    float lensq = Ve.x*Ve.x + Ve.y*Ve.y;
    float3 T1h = (lensq > 0.0f)
               ? float3(-Ve.y, Ve.x, 0.0f) * rsqrt(lensq)
               : float3(1.0f, 0.0f, 0.0f);
    float3 T2h = cross(Ve, T1h);

    // sample disk and warp
    float U1  = RandomFloat(seed);
    float U2  = RandomFloat(seed);
    float r   = sqrt(U1);
    float phi = 2.0f * PI * U2;
    float t1  = r * cos(phi);
    float t2  = r * sin(phi);
    float s   = 0.5f * (1.0f + Ve.z);
    t2 = (1.0f - s) * sqrt(saturate(1.0f - t1*t1)) + s * t2;

    // reprojection step
    float3 Nh = t1*T1h + t2*T2h + sqrt(saturate(1.0f - t1*t1 - t2*t2)) * Ve;
    float3 Ne = float3(alpha_x * Nh.x, alpha_y * Nh.y, max(0.0f, Nh.z));

    // convert to world space and return H
    return normalize(Ne.x * T1 + Ne.y * T2 + Ne.z * N);
}