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
    //return v0;
}


inline float3 SchlickFresnel(float3 F0, float cosTheta)
{
    return saturate(F0 + (1.0f - F0) * pow(abs(1.0f - cosTheta), 5.0f));
}

inline float D_GGX(float NdotH, float alpha)
{
    float alpha2 = alpha * alpha;
    float NdotH2 = NdotH * NdotH;

    float denom = (NdotH2 * (alpha2 - 1.0f) + 1.0f);
    denom = denom;
    return alpha2 / (PI * denom * denom);
}

// Smith Geometry functions for GGX
inline float G2_SmithGGX(float NdotV, float NdotL, float alpha)
{
    float alpha2 = alpha * alpha;

    float denomA = NdotV * sqrt(alpha2 + (1.0f - alpha2) * NdotL * NdotL);
    float denomB = NdotL * sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV);

    return 2.0f * NdotL * NdotV / (denomA + denomB);
}

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

// VNDF sampling GGX
// TODO: verify that this is correct
inline void SampleBRDF_GGX(
    uint mID,
    float3  outgoing,
    float3  normal,
    float3  flatNormal,
    inout float3 sample,
    float3  worldOrigin,
    inout uint2 seed)
{
    float alpha = materials[mID].Pr_Pm_Ps_Pc.x * materials[mID].Pr_Pm_Ps_Pc.x;

    // world to local transform
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);
    float3 T1, T2;
    CoordinateSystem(N, T1, T2);

    // hemisphere config
    float alpha_x = alpha;
    float alpha_y = alpha;
    float vx = dot(T1, V);
    float vy = dot(T2, V);
    float vz = dot(N,  V);

    float3 Ve = normalize(float3(alpha_x * vx,
                                 alpha_y * vy,
                                 vz));

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
    float3 Nh = t1*T1h + t2*T2h
              + sqrt(saturate(1.0f - t1*t1 - t2*t2)) * Ve;

    float3 Ne = float3(alpha_x * Nh.x,
                       alpha_y * Nh.y,
                       max(0.0f, Nh.z));
    Ne = normalize(Ne);

    // convert to world space
    float3 H = Ne.x * T1 + Ne.y * T2 + Ne.z * N;

    sample = reflect(-V, H);

    if(dot(sample, normal) < 0.0f)
        sample = float3(0,0,0);
}



// Evaluate the GGX BRDF for the given material
inline float3 EvaluateBRDF_GGX(uint mID, float3 normal, float3 incoming, float3 outgoing)
{
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);   // View direction
    float3 L = normalize(-incoming);  // Light direction
    float3 H = normalize(V + L);
    float NdotV = dot(N, V);
    float NdotL = dot(N, L);
    float NdotH = dot(N, H);
    float VdotH = dot(V, H);

    float3 F = SchlickFresnel(materials[mID].Ks, VdotH);
    float alpha = materials[mID].Pr_Pm_Ps_Pc.x * materials[mID].Pr_Pm_Ps_Pc.x;
    float D = D_GGX(NdotH, alpha);
    float G = G2_SmithGGX(NdotV, NdotL, alpha);

    // Specular BRDF
    float denominator = 4.0f * NdotV * NdotL;
    if(denominator < EPSILON)
        return float3(0,0,0);

    float3 specular = (F * D * G) / denominator;

    //Multiscatter GGX
    float Ess = ESS_LUT(mID, NdotV);
    float kms = (1.0f - Ess) / Ess;

    float3 specular_ess = specular * (1.0f + materials[mID].Ks * kms);

    if(any(isnan(specular_ess)) || any(isinf(specular_ess)))
        return float3(0,0,0);

    return specular_ess;
}

// Calculate the PDF for a given sample direction using GGX
inline float BRDF_PDF_GGX(uint mID, float3 normal, float3 incoming, float3 outgoing)
{
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);   // View direction
    float3 L = normalize(-incoming);  // Light direction
    float3 H = normalize(V + L);
    float NdotH = dot(N, H);
    float NdotV = dot(N, V);
    //float VdotH = dot(H, V);

    float alpha = materials[mID].Pr_Pm_Ps_Pc.x * materials[mID].Pr_Pm_Ps_Pc.x;
    float G1 = G1_SmithGGX(NdotV, alpha);
    float D = D_GGX(NdotH, alpha);

    return G1 * D / max(NdotV * 4.0f, EPSILON);
}
