#ifndef SUN_LIGHT_HLSLI
#define SUN_LIGHT_HLSLI

// --- Config ---
#define SUN_DIR_NORM        normalize(float3(0.3f, 1.0f, 0.1f)) // Matches your previous ddir
#define SUN_ANGULAR_DEG     0.53f
#define SUN_INTENSITY_VAL   10.0f
#define SUN_COLOR_VAL       float3(1.0f, 0.95f, 0.9f)
#define SUN_DIST_INF        1e7f // Effectively infinity

#ifndef PI
#define PI 3.14159265359
#endif

// --- Structs ---
struct SunSampleResult {
    float3 direction;
    float3 radiance;
    float  pdf;
    float  dist;
};

// --- Helpers ---
float GetSunSolidAngle(float thetaRad) {
    float h = sin(thetaRad * 0.5f);
    return 2.0f * PI * h * h;
}

void GetOrthoBasis(float3 N, out float3 T, out float3 B) {
    float3 Up = abs(N.z) < 0.999f ? float3(0, 0, 1) : float3(1, 0, 0);
    T = normalize(cross(Up, N));
    B = cross(N, T);
}

// --- API ---

// 1. NEE Sampling
SunSampleResult SampleSun(float2 u) {
    SunSampleResult r;
    
    // Convert degrees to radians (half angle)
    float thetaMax = (SUN_ANGULAR_DEG * 0.5f) * (PI / 180.0f);
    float cosThetaMax = cos(thetaMax);
    
    // Uniform cone sampling
    float z = 1.0f - u.y * (1.0f - cosThetaMax);
    float sinTheta = sqrt(max(0.0f, 1.0f - z * z));
    float phi = 2.0f * PI * u.x;
    
    float3 localDir = float3(cos(phi) * sinTheta, sin(phi) * sinTheta, z);
    
    float3 T, B;
    GetOrthoBasis(SUN_DIR_NORM, T, B);
    
    r.direction = normalize(localDir.x * T + localDir.y * B + localDir.z * SUN_DIR_NORM);
    
    float solidAngle = GetSunSolidAngle(thetaMax);
    r.pdf = 1.0f / solidAngle;
    r.radiance = (SUN_COLOR_VAL * SUN_INTENSITY_VAL) / solidAngle;
    r.dist = SUN_DIST_INF;
    
    return r;
}

// 2. MIS PDF ("As If" we sampled the light)
float GetSunPdf(float3 rayDir) {
    float thetaMax = (SUN_ANGULAR_DEG * 0.5f) * (PI / 180.0f);
    float cosThetaMax = cos(thetaMax);
    
    if (dot(rayDir, SUN_DIR_NORM) >= cosThetaMax) {
        return 1.0f / GetSunSolidAngle(thetaMax);
    }
    return 0.0f;
}

// 3. Evaluation (Hit Sky)
float3 EvaluateSun(float3 rayDir) {
    float thetaMax = (SUN_ANGULAR_DEG * 0.5f) * (PI / 180.0f);
    float cosThetaMax = cos(thetaMax);
    
    if (dot(rayDir, SUN_DIR_NORM) >= cosThetaMax) {
        float solidAngle = GetSunSolidAngle(thetaMax);
        return (SUN_COLOR_VAL * SUN_INTENSITY_VAL) / solidAngle;
    }
    return float3(0,0,0);
}

#endif