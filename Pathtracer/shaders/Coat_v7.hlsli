// The coat code mostly copies the GGX lobe except it doesnt use the energy compensation

// Orthonormal basis
inline void COAT_CoordinateSystem(float3 N, out float3 T, out float3 B)
{
    if (abs(N.z) < 0.999f) T = normalize(cross(float3(0.0f, 0.0f, 1.0f), N));
    else                   T = normalize(cross(float3(1.0f, 0.0f, 0.0f), N));
    B = cross(N, T);
}

// Schlick Fresnel
inline float3 COAT_SchlickFresnel(float3 F0, float cosTheta)
{
    return saturate(F0 + (1.0f - F0) * pow(abs(1.0f - cosTheta), 5.0f));
}

// GGX/Trowbridge-Reitz NDF
inline float COAT_D_GGX(float NdotH, float alpha)
{
    float a2    = alpha * alpha;
    float nh2   = NdotH * NdotH;
    float denom = nh2 * (a2 - 1.0f) + 1.0f;
    return a2 / (PI * denom * denom);
}

// Smith GGX masking (G1 and G2)
inline float COAT_G1_SmithGGX(float NdotV, float alpha)
{
    float a2    = alpha * alpha;
    float denom = sqrt(a2 + (1.0f - a2) * NdotV * NdotV) + NdotV;
    return (2.0f * NdotV) / denom;
}

inline float COAT_G2_SmithGGX(float NdotV, float NdotL, float alpha)
{
    return COAT_G1_SmithGGX(NdotV, alpha) * COAT_G1_SmithGGX(NdotL, alpha);
}

// VNDF GGX sampling
inline void SampleBRDF_COAT(
    uint    mID,
    float3  outgoing,
    float3  normal,
    float3  flatNormal,   // unused
    inout float3 sample,
    float3  worldOrigin,  // unused
    inout uint2 seed)
{
    float rough = saturate(materials[mID].Pcr_aniso_anisor.x); // coat roughness
    float alpha = max(EPSILON, rough * rough);

    float3 N = normalize(normal);
    float3 V = normalize(outgoing);
    float3 T1, T2;
    COAT_CoordinateSystem(N, T1, T2);

    // Stretch view
    float vx = dot(T1, V);
    float vy = dot(T2, V);
    float vz = dot(N,  V);
    float3 Ve = normalize(float3(alpha * vx, alpha * vy, vz));

    // Build basis around Ve
    float lensq = Ve.x * Ve.x + Ve.y * Ve.y;
    float3 T1h  = (lensq > 0.0f) ? float3(-Ve.y, Ve.x, 0.0f) * rsqrt(lensq)
                                 : float3(1.0f, 0.0f, 0.0f);
    float3 T2h  = cross(Ve, T1h);

    // Sample unit disk
    float U1  = RandomFloat(seed);
    float U2  = RandomFloat(seed);
    float r   = sqrt(U1);
    float phi = 2.0f * PI * U2;
    float t1  = r * cos(phi);
    float t2  = r * sin(phi);

    // Project back to hemisphere
    float s = 0.5f * (1.0f + Ve.z);
    t2      = (1.0f - s) * sqrt(saturate(1.0f - t1 * t1)) + s * t2;

    float3 Nh = t1 * T1h + t2 * T2h + sqrt(saturate(1.0f - t1 * t1 - t2 * t2)) * Ve;

    // Unstretch
    float3 Ne = normalize(float3(alpha * Nh.x, alpha * Nh.y, max(0.0f, Nh.z)));

    // Back to world, reflect
    float3 H = Ne.x * T1 + Ne.y * T2 + Ne.z * N;
    sample   = reflect(-V, H);

    if (dot(sample, normal) <= 0.0f)
        sample = float3(0, 0, 0);
}

inline float3 EvaluateBRDF_COAT(uint mID, float3 normal, float3 incoming, float3 outgoing, out float transmittance)
{
    if(dot(normal, -incoming) < 0.0f)
        return (float3).0f;
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);
    float3 L = normalize(-incoming);

    float NdotV = max(0.0f, dot(N, V));
    float NdotL = max(0.0f, dot(N, L));
    if (NdotV <= 0.0f || NdotL <= 0.0f) { transmittance = 1.0f; return 0.0.xxx; }

    float3 H    = normalize(V + L);
    float NdotH = max(0.0f, dot(N, H));
    float VdotH = max(0.0f, dot(V, H));

    // Fixed achromatic clearcoat F0 = 0.04
    const float3 F0 = 0.04.xxx;
    float3 F        = COAT_SchlickFresnel(F0, VdotH);

    float rough = saturate(materials[mID].Pcr_aniso_anisor.x);
    float alpha = max(EPSILON, rough * rough);

    float  D = COAT_D_GGX(NdotH, alpha);
    float  G = COAT_G2_SmithGGX(NdotV, NdotL, alpha);
    float denom = max(4.0f * NdotV * NdotL, EPSILON);

    float pc = saturate(materials[mID].Pr_Pm_Ps_Pc.w);

    float3 spec = pc * (F * D * G) / denom;

    float F0_diel = 0.04f;
    float Favg    = F0_diel + (1.0f - F0_diel) * (1.0f/21.0f);
    transmittance = saturate(1.0f - pc * Favg);

    return (any(isnan(spec)) || any(isinf(spec))) ? 0.0.xxx : spec;
}

inline float BRDF_PDF_COAT(uint mID, float3 N, float3 wi, float3 wo)
{
    if(dot(N, -wi) < 0.0f)
        return (float3).0f;
    float3 V = normalize(wo);
    float3 L = normalize(-wi);

    float NdotV = max(0.0f, dot(N, V));
    float NdotL = max(0.0f, dot(N, L));
    if (NdotV <= 0.0f || NdotL <= 0.0f) return 0.0f;

    float3 H = normalize(V + L);

    float  rough = saturate(materials[mID].Pcr_aniso_anisor.x);
    float  alpha = max(EPSILON, rough * rough);
    float  D     = COAT_D_GGX(max(0.0f, dot(N, H)), alpha);
    float  G1    = COAT_G1_SmithGGX(NdotV, alpha);

    return (D * G1) / max(4.0f * NdotV, EPSILON);
}