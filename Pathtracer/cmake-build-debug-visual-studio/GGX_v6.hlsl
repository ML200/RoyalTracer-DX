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
    denom = denom;
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

// ------------------------------------------------------------------------------
// SampleBRDF_GGX: samples a microfacet normal (half-vector) from Heitz’s VNDF
// and returns the *reflected* direction in `sample`. Uses alpha_x=alpha_y for
// isotropic GGX.  Adapts Heitz’s sampleGGXVNDF from the paper.
//
// Inputs:
//    mat.Pr_Pm_Ps_Pc.x = roughness parameter "alpha"
//    outgoing          = view (or outgoing) direction, in world space
//    normal            = macroscopic surface normal, in world space
//    flatNormal        = normal used for small bias offset
//    seed              = RNG state for random floats
// Outputs:
//    sample            = the *reflected* direction in world space
//    origin            = slightly bumped shading origin (optional)
// ------------------------------------------------------------------------------
inline void SampleBRDF_GGX(
    MaterialOptimized mat,
    float3  outgoing,
    float3  normal,
    float3  flatNormal,
    inout float3 sample,
    inout float3 origin,
    float3  worldOrigin,
    inout uint2 seed)
{
    // 1) Compute alpha^2 if your material encodes roughness that way
    float alpha = mat.Pr_Pm_Ps_Pc.x * mat.Pr_Pm_Ps_Pc.x;

    // 2) Set up world->local transform where "normal" is the local Z
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);
    float3 T1, T2;
    CoordinateSystem(N, T1, T2);

    // 3) Section 3.2: Transform view dir V -> Ve in "hemisphere config"
    //    Ve = normalize( (alpha_x * V.x), (alpha_y * V.y), V.z ) in local coords
    float alpha_x = alpha;
    float alpha_y = alpha;

    // Local coords of V relative to N,T1,T2:
    float vx = dot(T1, V);
    float vy = dot(T2, V);
    float vz = dot(N,  V);

    float3 Ve = normalize(float3(alpha_x * vx,
                                 alpha_y * vy,
                                 vz));

    // 4) Section 4.1: Build orthonormal basis around Ve
    float lensq = Ve.x*Ve.x + Ve.y*Ve.y;
    float3 T1h = (lensq > 0.0f)
               ? float3(-Ve.y, Ve.x, 0.0f) * rsqrt(lensq)
               : float3(1.0f, 0.0f, 0.0f);
    float3 T2h = cross(Ve, T1h);

    // 5) Section 4.2: sample disk & warp
    float U1  = RandomFloat(seed);
    float U2  = RandomFloat(seed);
    float r   = sqrt(U1);
    float phi = 2.0f * PI * U2;
    float t1  = r * cos(phi);
    float t2  = r * sin(phi);

    // "warp" step
    float s   = 0.5f * (1.0f + Ve.z);
    t2 = (1.0f - s) * sqrt(saturate(1.0f - t1*t1)) + s * t2;

    // 6) Section 4.3: reproject onto hemisphere
    float3 Nh = t1*T1h + t2*T2h
              + sqrt(saturate(1.0f - t1*t1 - t2*t2)) * Ve;

    // 7) Section 3.4: transform normal back to "ellipsoid" -> final half‐vector
    float3 Ne = float3(alpha_x * Nh.x,
                       alpha_y * Nh.y,
                       max(0.0f, Nh.z));
    Ne = normalize(Ne);

    // 8) Convert half‐vector from local coords to world space
    //    (unrotate by the same transform that took N->(0,0,1))
    float3 H = Ne.x * T1 + Ne.y * T2 + Ne.z * N;

    // 9) Reflect view about H to get the *sampled* direction
    //    The built‐in function reflect(I,N) = I - 2 (N·I) N
    //    We want L so that V + L is “2H * (some factor)”
    sample = reflect(-V, H);

    // 10) Shift origin to avoid self‐intersection
    origin = worldOrigin + s_bias * flatNormal;
}



// Evaluate the GGX BRDF for the given material
inline float3 EvaluateBRDF_GGX(MaterialOptimized mat, float3 normal, float3 incoming, float3 outgoing)
{
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);   // View direction
    float3 L = normalize(-incoming);  // Light direction
    float3 H = normalize(V + L);
    float NdotV = dot(N, V);
    float NdotL = dot(N, L);
    float NdotH = dot(N, H);
    float VdotH = dot(V, H);

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
    float NdotH = dot(N, H);
    float NdotV = dot(N, V);
    float VdotH = dot(H, V);

    float alpha = mat.Pr_Pm_Ps_Pc.x * mat.Pr_Pm_Ps_Pc.x;
    float G1 = G1_SmithGGX(NdotV, alpha);
    float D = D_GGX(NdotH, mat.Pr_Pm_Ps_Pc.x);

    return G1 * D / (NdotV * 4.0f);
}
