// Simple Cosine-Weighted hemisphere sampling
float3 RandomUnitVectorInHemisphere(float3 normal, inout uint2 seed)
{
    // Generate two random numbers
    float u1 = RandomFloat(seed);
    float u2 = RandomFloat(seed);

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
    if (dot(hemisphereSample, normal) < 0.0f) {
        hemisphereSample = -hemisphereSample;
    }

    return hemisphereSample;
}



// Sample the BRDF of the given material
void SampleBRDF_Lambertian(uint mID, float3 incoming, float3 normal, float3 flatNormal, inout float3 sample, float3 worldOrigin, inout uint2 seed) {
    // Sample a random direction in the hemisphere oriented around the flatNormal
    sample = RandomUnitVectorInHemisphere(normal, seed);
}

// Evaluate the BRDF for the given material
float3 EvaluateBRDF_Lambertian(uint mID, float3 normal, float3 incoming, float3 outgoing, out float transmittance) {
    transmittance = 1.0f - materials[mID].Kd.w;
    // For Lambertian reflection, the BRDF is constant
    // BRDF = Kd / PI
    return materials[mID].Kd.xyz / PI;
}

// Calculate the PDF for a given sample direction
float BRDF_PDF_Lambertian(uint mID, float3 normal, float3 incoming, float3 outgoing) {
    // For cosine-weighted hemisphere sampling over a Lambertian surface
    return max(dot(normalize(normal), normalize(-incoming)), 0.0f) / PI;
}

