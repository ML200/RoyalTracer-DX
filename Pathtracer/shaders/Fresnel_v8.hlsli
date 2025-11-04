static inline float3 ComputeF0(int mID, float eta_i, float eta_t)
{
    float  R0s = (eta_i - eta_t) / (eta_i + eta_t);
    R0s *= R0s;
    float3 R0 = R0s.xxx;

    float3 Kd = saturate(materials[mID].Kd.xyz);
    float  Pm = saturate(materials[mID].Pr_Pm_Ps_Pc.y);

    return lerp(R0, Kd, Pm);
}

static inline float3 FresnelSchlick(float3 F0, float3 wo, float3 n, float eta_i, float eta_t)
{
    float ci = saturate(abs(dot(wo, n)));

    float oneMinus  = 1.0f - ci;
    float oneMinus2 = oneMinus * oneMinus;
    float oneMinus5 = oneMinus2 * oneMinus2 * oneMinus;

    return F0 + (1.0f - F0) * oneMinus5;
}

static inline float3 FresnelSchlickTIR(float3 F0, float3 wo, float3 n, float eta_i, float eta_t)
{
    float ci = saturate(abs(dot(wo, n)));

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

    return F0 + (1.0f - F0) * oneMinus5;
}