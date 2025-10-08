inline float Avg3_TRANS(float3 v) { return (v.x + v.y + v.z) * (1.0f/3.0f); }

inline float IORtoF0_TRANS(float ior)
{
    float a = (ior - 1.0f) / (ior + 1.0f);
    return a * a;
}

inline float3 ComputeBaseF0_TRANS(uint mID)
{
    float  metallic  = materials[mID].Pr_Pm_Ps_Pc.y;
    float3 baseColor = materials[mID].Kd.xyz;
    float  ior       = materials[mID].Ni;

    float  F0_diel  = IORtoF0_TRANS(ior);             // dielectric F0
    float3 F0_metal = baseColor;                      // colored F0 for metals
    return saturate(lerp(F0_diel.xxx, F0_metal, metallic));
}

// Schlick scalar for dielectrics using the two-sided IoR pair
inline float FresnelSchlickIor_TRANS(float cosThetaI, float etaI, float etaT)
{
    cosThetaI = saturate(abs(cosThetaI));
    // R0 from the interface (symmetrical for either direction)
    float r0 = (etaT - etaI) / (etaT + etaI);
    r0 = r0 * r0;

    float m = 1.0f - cosThetaI;
    float schlick = r0 + (1.0f - r0) * (m*m*m*m*m);
    return saturate(schlick);
}

// ---------- GGX NDF & Smith G ----------
inline float D_GGX_TRANS(float NdotH, float alpha)
{
    float a2    = alpha * alpha;
    float NdotH2 = NdotH * NdotH;
    float denom  = (NdotH2 * (a2 - 1.0f) + 1.0f);
    return a2 / (PI * denom * denom);
}

inline float G1_SmithGGX_TRANS(float NdotV, float alpha)
{
    float a2    = alpha * alpha;
    float denom = sqrt(a2 + (1.0f - a2) * NdotV * NdotV) + NdotV;
    return 2.0f * NdotV / denom;
}

inline float G2_SmithGGX_TRANS(float NdotV, float NdotL, float alpha)
{
    return G1_SmithGGX_TRANS(NdotV, alpha) * G1_SmithGGX_TRANS(NdotL, alpha);
}

inline void CoordinateSystem_TRANS(float3 N, out float3 T, out float3 B)
{
    if (abs(N.z) < 0.999f) T = normalize(cross(float3(0,0,1), N));
    else                   T = normalize(cross(float3(1,0,0), N));
    B = cross(N, T);
}

// Refract wo across normal n with ratio eta (side(wo) -> other side)
// Returns false if TIR occurs
inline bool RefractVector_TRANS(float3 wo, float3 n, float eta, out float3 wi)
{
    float coso = dot(n, wo);
    float k    = 1.0f - eta * eta * (1.0f - coso * coso);
    if (k < 0.0f) { wi = 0.0.xxx; return false; }
    wi = eta * (-wo) + (eta * coso - sqrt(k)) * n;
    return true;
}

// VNDF sampling (Heitz)
inline void SampleBTDF_GGX_TRANS(
    uint    mID,
    float3  outgoing,
    float3  normal,
    float3  flatNormal,
    inout float3 sample,
    float3  worldOrigin,
    inout uint2 seed)
{
    // GGX roughness
    float r     = materials[mID].Pr_Pm_Ps_Pc.x;
    float alpha = r * r;

    // Build local frame
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);
    float3 T1, T2;
    CoordinateSystem_TRANS(N, T1, T2);

    // Stretch view
    float alpha_x = alpha;
    float alpha_y = alpha;

    float vx = dot(T1, V);
    float vy = dot(T2, V);
    float vz = dot(N,  V);

    float3 Ve = normalize(float3(alpha_x * vx, alpha_y * vy, vz));

    // Orthonormal basis around Ve
    float lensq = Ve.x*Ve.x + Ve.y*Ve.y;
    float3 T1h = (lensq > 0.0f) ? float3(-Ve.y, Ve.x, 0.0f) * rsqrt(lensq)
                                : float3(1.0f, 0.0f, 0.0f);
    float3 T2h = cross(Ve, T1h);

    // Sample disk & warp
    float U1  = RandomFloat(seed);
    float U2  = RandomFloat(seed);
    float rU  = sqrt(U1);
    float phi = 2.0f * PI * U2;
    float t1  = rU * cos(phi);
    float t2  = rU * sin(phi);

    float s = 0.5f * (1.0f + Ve.z);
    t2 = (1.0f - s) * sqrt(saturate(1.0f - t1*t1)) + s * t2;

    // Reproject & unstretch to get the microfacet normal H (VNDF sample)
    float3 Nh = t1*T1h + t2*T2h + sqrt(saturate(1.0f - t1*t1 - t2*t2)) * Ve;
    float3 Ne = float3(alpha_x * Nh.x, alpha_y * Nh.y, max(0.0f, Nh.z));
    float3 H  = normalize(Ne.x * T1 + Ne.y * T2 + Ne.z * N);

    // Ensure H faces the same side as V for consistency
    if (dot(V, H) < 0.0f) H = -H;

    // IORs for refraction (air <-> material)
    float ior   = 2.5f;
    float NdotV = dot(N, V);
    bool  entering = (NdotV > 0.0f);
    float etaI = entering ? 1.0f : ior;
    float etaT = entering ? ior : 1.0f;
    float eta  = etaI / etaT;

    // Refract across the microfacet normal
    float3 L;
    if (!RefractVector_TRANS(V, H, eta, L))
    {
        // TIR for this microfacet: reject
        sample = 0.0.xxx;
        return;
    }

    // Must lie on the other hemisphere
    if (dot(N, L) * NdotV >= 0.0f)
    {
        sample = 0.0.xxx;
        return;
    }

    sample = normalize(L);
}

// ---------- Evaluate rough GGX BTDF ----------
inline float3 EvaluateBTDF_GGX_TRANS(uint mID, float3 normal, float3 incoming, float3 outgoing, out float transmittance)
{
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);     // outgoing (view)
    float3 L = normalize(-incoming);    // incoming from other side

    float NdotV = dot(N, V);
    float NdotL = dot(N, L);

    // Transmission only if directions are on opposite sides
    if (NdotV * NdotL >= 0.0f) return 0.0.xxx;

    float r     = materials[mID].Pr_Pm_Ps_Pc.x;
    float alpha = max(EPSILON, r * r);

    // Determine side and relative IORs
    float ior     = materials[mID].Ni;
    bool  entering = (NdotV > 0.0f);
    float etaI = entering ? 1.0f : ior;
    float etaT = entering ? ior : 1.0f;

    // Half vector for transmission (Walter 2007)
    float3 H = normalize(etaI * V + etaT * L);
    if (dot(N, H) < 0.0f) H = -H;

    float NdotH = max(0.0f, dot(N, H));
    float VdotH = dot(V, H);
    float LdotH = dot(L, H);

    // Exact dielectric Fresnel at the microfacet
    float  F = FresnelSchlickIor_TRANS(abs(VdotH), etaI, etaT);

    // GGX terms
    float  D = D_GGX_TRANS(NdotH, alpha);
    float  G = G2_SmithGGX_TRANS(abs(NdotV), abs(NdotL), alpha);

    // Walter 2007 BTDF (Eq. 21 form) for rough refraction
    // ft = (1 - F) * D * G * |(V·H)(L·H)| * η_t^2 / (|N·V||N·L| * (η_i(V·H) + η_t(L·H))^2)
    float  denom = (etaI * VdotH + etaT * LdotH);
    float  denom2 = max(denom * denom, EPSILON);
    float  scale  = (1.0f - F) * D * G * (etaT * etaT)
                    * abs(VdotH * LdotH)
                    / (max(abs(NdotV * NdotL), EPSILON) * denom2);

    float3 ft = scale.xxx;

    // float3 glassTint = 1.0.xxx; // TODO: tint/absorption
    // ft *= glassTint;

    return (any(isnan(ft)) || any(isinf(ft))) ? 0.0.xxx : ft;
}

inline float BTDF_PDF_GGX_TRANS(uint mID, float3 N, float3 wi, float3 wo)
{
    float3 V = normalize(wo);
    float3 L = normalize(-wi);

    float NdotV = dot(N, V);
    float NdotL = dot(N, L);

    // Transmission only
    if (NdotV * NdotL >= 0.0f) return 0.0f;

    float r     = materials[mID].Pr_Pm_Ps_Pc.x;
    float alpha = max(EPSILON, r * r);

    float ior     = 2.5f;
    bool  entering = (NdotV > 0.0f);
    float etaI = entering ? 1.0f : ior;
    float etaT = entering ? ior : 1.0f;
    float eta  = etaI / etaT;

    // Transmission half-vector
    float3 H = normalize(etaI * V + etaT * L);
    if (dot(N, H) < 0.0f) H = -H;

    float NdotH = max(0.0f, dot(N, H));
    float VdotH = dot(V, H);
    float LdotH = dot(L, H);

    // VNDF half-vector PDF term
    float  D    = D_GGX_TRANS(NdotH, alpha);
    float  G1   = G1_SmithGGX_TRANS(abs(NdotV), alpha);
    float  pdf_m = D * G1 * abs(VdotH) / max(abs(NdotV), EPSILON);

    // Refraction Jacobian
    float  denom = (VdotH + eta * LdotH);
    float  dwh_dwi = abs((eta * eta * LdotH) / max(denom * denom, EPSILON));

    float  pdf = pdf_m * dwh_dwi;
    return (isnan(pdf) || isinf(pdf)) ? 0.0f : pdf;
}