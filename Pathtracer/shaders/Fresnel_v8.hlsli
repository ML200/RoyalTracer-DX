static inline float3 ComputeF0Dielectric(float eta_i, float eta_t)
{
    float  R0s = (eta_i - eta_t) / (eta_i + eta_t);
    R0s *= R0s;
    return R0s.xxx;
}

// Only dielectric part of the fresnel equation - single float
static inline float3 FresnelDielectric(float3 wo, float3 n, float eta_i, float eta_t)
{
    float ci = abs(dot(wo, n));
    float3 R0 = ComputeF0Dielectric(eta_i, eta_t);

    float oneMinus  = 1.0f - ci;
    float oneMinus2 = oneMinus * oneMinus;
    float oneMinus5 = oneMinus2 * oneMinus2 * oneMinus;

    return R0 + (1.0f - R0) * oneMinus5;
}

static inline float3 FresnelDielectricTIR(float3 wo, float3 n, float eta_i, float eta_t)
{
    float ci = abs(dot(wo, n));
    // Exact TIR test with provided indices
    float eta   = eta_i / eta_t;
    float sin2I = max(0.0f, 1.0f - ci * ci);
    if (eta * eta * sin2I > 1.0f)
    {
        return 1.0f.xxx; // total internal reflection
    }

    float oneMinus  = 1.0f - ci;
    float oneMinus2 = oneMinus * oneMinus;
    float oneMinus5 = oneMinus2 * oneMinus2 * oneMinus;

    float3 R0 = ComputeF0Dielectric(eta_i, eta_t);

    return R0 + (1.0f - R0) * oneMinus5;
}

static inline float3 FresnelConductor(float3 F0, float3 wo, float3 n)
{
    float ci = saturate(abs(dot(wo, n)));

    float oneMinus  = 1.0f - ci;
    float oneMinus2 = oneMinus * oneMinus;
    float oneMinus5 = oneMinus2 * oneMinus2 * oneMinus;

    return F0 + (1.0f - F0) * oneMinus5;
}