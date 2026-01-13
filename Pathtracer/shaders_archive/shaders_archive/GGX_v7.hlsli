// IoR stuff
inline float Avg3(float3 v) { return (v.x + v.y + v.z) * (1.0/3.0); }
inline float IORtoF0(float ior) {
    float a = (ior - 1.0f) / (ior + 1.0f);
    return a * a;
}

inline float3 ComputeBaseF0(uint mID) {
    float  metallic  = materials[mID].Pr_Pm_Ps_Pc.y;
    float3 baseColor = materials[mID].Kd.xyz;
    float  ior      =  materials[mID].Ni;

    float  F0_diel  = IORtoF0(ior);                     // achromatic dielectric F0
    float3 F0_metal = baseColor;                        // colored F0 for metals
    return saturate(lerp(F0_diel.xxx, F0_metal, metallic)); // Blend baby
}


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
/*inline float G2_SmithGGX(float NdotV, float NdotL, float alpha)
{
    float alpha2 = alpha * alpha;

    float denomA = NdotV * sqrt(alpha2 + (1.0f - alpha2) * NdotL * NdotL);
    float denomB = NdotL * sqrt(alpha2 + (1.0f - alpha2) * NdotV * NdotV);

    return 2.0f * NdotL * NdotV / (denomA + denomB);
}*/

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

// VNDF sampling GGX
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

    if(dot(sample, normal) <= 0.0f)
        sample = float3(0,0,0);
}



// Evaluate the GGX BRDF for the given material
inline float3 EvaluateBRDF_GGX(uint mID, float3 normal, float3 incoming, float3 outgoing, out float transmittance)
{
    if(dot(normal, -incoming) < 0.0f)
        return (float3).0f;
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);
    float3 L = normalize(-incoming);

    float NdotV = max(0.0f, dot(N, V));
    float NdotL = max(0.0f, dot(N, L));
    if (NdotV <= 0.0f || NdotL <= 0.0f) return 0.0.xxx;

    float3 H = normalize(V + L);
    float NdotH = max(0.0f, dot(N, H));
    float VdotH = max(0.0f, dot(V, H));

    float3 F0 = ComputeBaseF0(mID);
    float3 F  = SchlickFresnel(F0, VdotH);

    float alpha = materials[mID].Pr_Pm_Ps_Pc.x;
    alpha *= alpha; // alpha = roughness^2

    float D = D_GGX(NdotH, alpha);
    float G = G2_SmithGGX(NdotV, NdotL, alpha);

    float denom = max(4.0f * NdotV * NdotL, EPSILON);
    float3 specular = (F * D * G) / denom;

    // Multiscatter compensation
    float Ess  = ESS_LUT(mID, NdotV);
    float kms  = (1.0f - Ess) / max(Ess, EPSILON);
    float3 specular_ess = specular * (1.0f + Avg3(F0) * kms);

    // dielectric F0 (scalar) -> hemisphere average
    float  F0_diel = IORtoF0(materials[mID].Ni);
    float  Favg    = F0_diel + (1.0f - F0_diel) * (1.0f/21.0f);
    transmittance = (1.0f - materials[mID].Pr_Pm_Ps_Pc.y) * (1.0f - Favg);

    return (any(isnan(specular_ess)) || any(isinf(specular_ess))) ? 0.0.xxx : specular_ess;
}


// Calculate the PDF for a given sample direction using GGX
inline float BRDF_PDF_GGX(uint mID, float3 N, float3 wi, float3 wo)
{
    if(dot(N, -wi) < 0.0f)
        return (float3).0f;
    float3 V = normalize(wo);
    float3 L = normalize(-wi);
    float  NdotV = max(0.0f, dot(N, V));
    float  NdotL = max(0.0f, dot(N, L));
    if (NdotV <= 0.0f || NdotL <= 0.0f) return 0.0f;

    float3 H = normalize(V + L);

    float  r = materials[mID].Pr_Pm_Ps_Pc.x;
    float  alpha = max(EPSILON, r*r);
    float  D  = D_GGX(max(0.0f, dot(N, H)), alpha);
    float  G1 = G1_SmithGGX(NdotV, alpha);

    return (D * G1) / max(4.0f * NdotV, EPSILON);
}

