float FresnelExact(float3 wo, float3 n, float eta_i, float eta_t)
{
    float cosI = dot(wo, n);
    if (cosI < 0.0f) return -1.0f; // invalid input

    float eta = eta_i / eta_t;

    float sin2I = 1.0f - cosI * cosI;
    float sin2T = eta * eta * sin2I;

    // Total Internal Reflection: all light is reflected.
    if (sin2T > 1.0f)
    {
        return 1.0f;
    }

    float cosT = sqrt(max(0.0f, 1.0f - sin2T));

    // Fresnel term for s-polarization (perpendicular)
    float term1_s = eta_i * cosI;
    float term2_s = eta_t * cosT;
    float Rs = (term1_s - term2_s) / (term1_s + term2_s);

    // Fresnel term for p-polarization (parallel)
    float term1_p = eta_t * cosI;
    float term2_p = eta_i * cosT;
    float Rp = (term1_p - term2_p) / (term1_p + term2_p);

    // For unpolarized light, the reflectance is the average of the two
    return 0.5f * (Rs * Rs + Rp * Rp);
}

float FresnelSchlick(float3 wo, float3 n, float eta_i, float eta_t)
{
    float cosI = dot(wo, n);
    if (cosI < 0.0f) return -1.0f;

    // This optional TIR guard makes the Schlick approximation exact for TIR cases
    float eta = eta_i / eta_t;
    float sin2I = 1.0f - cosI * cosI;
    if (eta * eta * sin2I > 1.0f)
    {
        return 1.0f;
    }

    // Reflectance at normal incidence (0 degrees)
    float R0 = (eta_i - eta_t) / (eta_i + eta_t);
    R0 = R0 * R0;

    // Schlick's approximation
    float m  = 1.0f - cosI;
    float m2 = m * m;
    float m5 = m2 * m2 * m; // pow(m, 5)

    return R0 + (1.0f - R0) * m5;
}