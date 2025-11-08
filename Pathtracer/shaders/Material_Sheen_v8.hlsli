// Constants for now
static const float  SHEEN_R      = 0.10f;
static const float3 SHEEN_COLOR  = 1.0.xxx;

// Get the transmitted energy
inline float SheenAlpha_FromLUT(uint mID, float NdotV)
{
    NdotV = saturate(NdotV);
    float f = NdotV * (16 - 1);
    int   i0 = (int)floor(f);
    int   i1 = min(i0 + 1, 16 - 1);
    float w  = f - i0;

    float a0 = materials[mID].SheenLUT[i0];
    float a1 = materials[mID].SheenLUT[i1];
    return lerp(a0, a1, w);
}


// D term (Eq. 2): D(m) = (2 + 1/r) * sin(theta_h)^(1/r) / (2pi)
inline float SHEEN_D_Charlie(float NdotH)
{
    float r      = SHEEN_R;
    float invr   = 1.0f / max(1e-4f, r);
    float sinTh2 = saturate(1.0f - NdotH * NdotH);
    float sinTh  = sqrt(sinTh2);
    float D      = (2.0f + invr) * pow(max(1e-8f, sinTh), invr) * (0.5f * INV_PI); // 1/(2pi)
    return D;
}

// Λ fit from the paper (Table 1 + Eq. 3), correlated Smith G = 1 / (1 + A(V) + A(L))
inline float SHEEN_LambdaFit_params(out float a, out float b, out float c, out float d, out float e)
{
    // interpolate between r=0.0 and r=1.0 fits: P = (1-r)^2 * P0 + (1-(1-r)^2) * P1
    float r   = saturate(SHEEN_R);
    float w0  = (1.0f - r); w0 *= w0;
    float w1  = 1.0f - w0;

    // r = 0.0   :  a=25.3245 b=3.32435 c=0.16801 d=-1.27393 e=-4.85967
    // r = 1.0   :  a=21.5473 b=3.82987 c=0.19823 d=-1.97760 e=-4.32054
    float a0=25.3245f, b0=3.32435f, c0=0.16801f, d0=-1.27393f, e0=-4.85967f;
    float a1=21.5473f, b1=3.82987f, c1=0.19823f, d1=-1.97760f, e1=-4.32054f;

    a = w0*a0 + w1*a1;
    b = w0*b0 + w1*b1;
    c = w0*c0 + w1*c1;
    d = w0*d0 + w1*d1;
    e = w0*e0 + w1*e1;

    return r;
}

inline float SHEEN_L_eval(float x, float a, float b, float c, float d, float e)
{
    return a / (1.0f + b * pow(max(1e-4f, x), c)) + d * x + e;
}

inline float SHEEN_Lambda_Charlie(float cosTheta)
{
    // piecewise A using L(x) = a/(1 + b x^c) + d x + e
    float a, b, c, d, e;
    (void)SHEEN_LambdaFit_params(a, b, c, d, e);

    float x = saturate(cosTheta);

    float Lx      = SHEEN_L_eval(x,        a, b, c, d, e);
    float L_half  = SHEEN_L_eval(0.5f,     a, b, c, d, e);
    float L_1mx   = SHEEN_L_eval(1.0f - x, a, b, c, d, e);

    // piecewise definition
    float val = (x < 0.5f) ? exp(Lx)
                           : exp(2.0f * L_half - L_1mx);
    return val;
}

inline float SHEEN_G_Charlie(float NdotV, float NdotL)
{
    float lambdaV = SHEEN_Lambda_Charlie(NdotV);
    float lambdaL = SHEEN_Lambda_Charlie(NdotL);
    return 1.0f / (1.0f + lambdaV + lambdaL); // correlated Smith
}

// Half-vector sampling for Charlie: sample m ~ D(m) * (N·m)
// Derivation gives sin^2(theta_h) = u^(2r/(2r+1))
inline float3 SHEEN_SampleHalfVector(uint2 seed, float3 N, out float NdotH, out float pdf_H)
{
    float u1  = RandomFloat(seed);
    float u2  = RandomFloat(seed);

    float r        = SHEEN_R;
    float expo     = (2.0f * r) / (2.0f * r + 1.0f);
    float sin2Th   = pow(saturate(u1), expo);
    float cosTh    = sqrt(saturate(1.0f - sin2Th));
    float sinTh    = sqrt(sin2Th);
    float phi      = 2.0f * PI * u2;

    float3 T, B;
    CoordinateSystem(N, T, B);
    float3 H = sinTh * cos(phi) * T + sinTh * sin(phi) * B + cosTh * N;
    H = normalize(H);

    NdotH = saturate(dot(N, H));

    // p_H(m) = D(m) * (N·m)
    pdf_H = SHEEN_D_Charlie(NdotH) * NdotH;
    return H;
}

inline void SampleBRDF_SHEEN(
    uint    mID,
    float3  outgoing,       // wo
    float3  normal,
    float3  flatNormal,
    inout float3 sample,
    inout uint2 seed)
{
    float3 N = normalize(normal);
    float3 V = normalize(outgoing);

    float NdotV = max(0.0f, dot(N, V));
    if (NdotV <= 0.0f) { sample = float3(0,0,0); return; }

    float NdotH, pdfH;
    float3 H = SHEEN_SampleHalfVector(seed, N, NdotH, pdfH);

    // reflect view about H to get incident direction
    float3 wi = reflect(-V, H);
    if (dot(N, wi) <= 0.0f) { sample = float3(0,0,0); return; }

    sample = wi; // convention: integrator will use L = normalize(-wi)
}

// BRDF eval: f = w * color * F * G * D / (4 N·V N·L); For sheen we take F~1
inline float3 EvaluateBRDF_SHEEN(uint mID, float3 normal, float3 incoming, float3 outgoing, out float transmittance)
{
    if(dot(normal, -incoming) < 0.0f)
        return (float3).0f;
    float w = saturate(materials[mID].Pr_Pm_Ps_Pc.z); // sheen weight
    if (w <= 0.0f) { transmittance = 1.0f; return 0.0.xxx; }

    float3 N = normalize(normal);
    float3 V = normalize(outgoing);
    float3 L = normalize(-incoming);

    float NdotV = max(0.0f, dot(N, V));
    float NdotL = max(0.0f, dot(N, L));
    if (NdotV <= 0.0f || NdotL <= 0.0f) { transmittance = 1.0f; return 0.0.xxx; }

    float aV = SheenAlpha_FromLUT(mID, NdotV); // reflectance seen along V
    float aL = SheenAlpha_FromLUT(mID, NdotL); // reflectance seen along L

    // Sheen tint
    float tintLum = saturate(Luma(SHEEN_COLOR));
    aV *= tintLum;
    aL *= tintLum;

    // Fraction of energy that gets through the sheen layer
    float T_in  = saturate(1.0f - w * aL);
    float T_out = saturate(1.0f - w * aV);

    transmittance = sqrt(T_in * T_out);
    float3 H    = normalize(V + L);
    float NdotH = max(0.0f, dot(N, H));

    float  D = SHEEN_D_Charlie(NdotH);
    float  G = SHEEN_G_Charlie(NdotV, NdotL);
    float  denom = max(1e-6f, 4.0f * NdotV * NdotL);

    float3 F = 1.0.xxx; // achromatic sheen; constant Fresnel
    return w * SHEEN_COLOR * (F * (G * D / denom));
}

inline float BRDF_PDF_SHEEN(uint mID, float3 N, float3 wi, float3 wo)
{
    if(dot(N, -wi) < 0.0f)
        return (float3).0f;
    float3 V = normalize(wo);
    float3 L = normalize(-wi);
    float NdotL = max(0.0f, dot(N, L));
    if (NdotL <= 0.0f) return 0.0f;

    float3 H    = normalize(V + L);
    float NdotH = max(0.0f, dot(N, H));
    float VoH   = max(1e-6f, dot(V, H));

    // p(wi) = p(H) / (4 |V·H|), with p(H) = D(H) * (N·H)
    return SHEEN_D_Charlie(NdotH) * NdotH / (4.0f * VoH);
}