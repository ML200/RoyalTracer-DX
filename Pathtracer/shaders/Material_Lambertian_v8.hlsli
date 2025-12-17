// Simple Cosine-Weighted hemisphere sampling
float3 CosineUnitVectorInHemisphere(float3 normal, inout uint seed)
{
    // Generate two random numbers
    float u1 = RandomFloatSingle(seed);
    float u2 = RandomFloatSingle(seed);

    float r = sqrt(u1);
    float theta = 2.0 * 3.14159265358979323846 * u2;

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

// Evaluate the BRDF for the given material
inline float3 EvaluateBRDF_Lambertian(uint mID, float3 normal, float3 flatNormal, float3 incoming, float3 outgoing, float etai, float etat, float3 Kd) {
    if(dot(-incoming, flatNormal) <= 0.0f)
        return float3(0,0,0);
    // For Lambertian reflection, the BRDF is constant
    return Kd / PI;
}

// No transmittance needed, lambertian is the lowest lobe
// Sampling weight function describes an approximation of the transmittance before sampling, as incoming isnt known yet
inline float Sampling_Weight_Lambertian(uint mID, float3 normal, float3 outgoing){
    return 1.0f;
}

// Sample the BRDF of the given material
inline float3 SampleBRDF_Lambertian(uint mID, float3 incoming, float3 normal, float3 flatNormal, inout uint seed) {
    // Sample a random direction in the hemisphere oriented around the flatNormal
    return CosineUnitVectorInHemisphere(normal, seed);
}

// Calculate the PDF for a given sample direction
inline float BRDF_PDF_Lambertian(uint mID, float3 normal, float3 flatNormal, float3 incoming, float3 outgoing) {
    if(dot(-incoming, flatNormal) <= 0.0f)
        return 0.0f;
    // For cosine-weighted hemisphere sampling over a Lambertian surface
    return max(dot(normalize(normal), normalize(-incoming)), 0.0f) / PI;
}

