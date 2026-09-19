float3 CosineUnitVectorInHemisphereFrom(float3 normal, float2 u)
{
    float r = sqrt(u.x);
    float theta = 2.0 * 3.14159265358979323846 * u.y;

    float x = r * cos(theta);
    float y = r * sin(theta);

    float z = sqrt(max(0.0f, 1.0f - x*x - y*y));

    float3 h = normal;
    float3 up = abs(normal.z) < 0.999 ? float3(0,0,1) : float3(1,0,0);
    float3 right = normalize(cross(up, h));
    float3 forward = cross(h, right);

    float3 hemisphereSample = x * right + y * forward + z * h;

    hemisphereSample = normalize(hemisphereSample);
    return hemisphereSample;
}

float3 CosineUnitVectorInHemisphere(float3 normal, inout uint seed)
{
    float u1 = RandomFloatSingle(seed);
    float u2 = RandomFloatSingle(seed);
    return CosineUnitVectorInHemisphereFrom(normal, float2(u1, u2));
}

// Energy-preserving Oren-Nayar (EON), with an analytic diffuse MS term.
inline float OrenNayarDirectionalAlbedo(float mu, float roughness)
{
    float x = 1.0f - saturate(mu);
    float g = x * (.0571085289f + x * (.491881867f + x * (-.332181442f + x * .0714429953f)));
    return (1.0f + roughness * g) / (1.0f + (.5f - 2.0f / (3.0f * PI)) * roughness);
}

inline float EvaluateOrenNayar(float3 N, float3 V, float3 L, float roughness)
{
    float v = max(dot(N, V), 0.0f), l = max(dot(N, L), 0.0f);
    if (v <= 0.0f || l <= 0.0f) return 0.0f;
    float s = dot(V, L) - v * l;
    float A = rcp(1.0f + (.5f - 2.0f / (3.0f * PI)) * roughness);
    float avg = A * (1.0f + (2.0f / 3.0f - 28.0f / (15.0f * PI)) * roughness);
    return A * INV_PI * (1.0f + roughness * (s > 0.0f ? s / max(v, l) : s)) +
           INV_PI * (1.0f - OrenNayarDirectionalAlbedo(v, roughness)) *
           (1.0f - OrenNayarDirectionalAlbedo(l, roughness)) / max(1.0f - avg, 1e-8f);
}

// Keep legacy entry points and cosine proposals for integrator/replay compatibility.
inline float3 EvaluateBRDF_Lambertian(uint mID, float3 normal, float3 flatNormal, float3 incoming, float3 outgoing, float etai, float etat, float3 Kd) {
    if(dot(-incoming, flatNormal) <= 0.0f)
        return float3(0,0,0);
    return Kd * EvaluateOrenNayar(normalize(normal), normalize(outgoing), normalize(-incoming),
                                LoadDiffuseRoughness(mID));
}

inline float Sampling_Weight_Lambertian(uint mID, float3 normal, float3 outgoing){
    return 1.0f;
}

// Sample the cosine-weighted diffuse hemisphere.
inline float3 SampleBRDF_Lambertian(uint mID, float3 incoming, float3 normal, float3 flatNormal, inout uint seed) {
    return CosineUnitVectorInHemisphere(normal, seed);
}

inline float BRDF_PDF_Lambertian(uint mID, float3 normal, float3 flatNormal, float3 incoming, float3 outgoing) {
    if(dot(-incoming, flatNormal) <= 0.0f)
        return 0.0f;
    return max(dot(normalize(normal), normalize(-incoming)), 0.0f) / PI;
}
