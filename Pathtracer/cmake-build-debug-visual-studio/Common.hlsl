// Hit information, aka ray payload
// This sample only carries a shading color and hit distance.
// Note that the payload should be kept as small as possible,
// and that its size must be declared in the corresponding
// D3D12_RAYTRACING_SHADER_CONFIG pipeline subobject.
struct HitInfo {
  float4 colorAndDistance;
  float3 emission;
  float3 direction;
  float3 origin;
  float4 util; //IMPORTANT: util info: miss flag, random1, random2, empty
};

// Attributes output by the raytracing when hitting a surface,
// here the barycentric coordinates
struct Attributes {
  float2 bary;
};


uint Hash(uint x) {
    x += (x << 10);
    x ^= (x >> 6);
    x += (x << 3);
    x ^= (x >> 11);
    x += (x << 15);
    return x;
}

uint TausStep(uint z, int S1, int S2, int S3, uint M) {
    uint b = (((z << S1) ^ z) >> S2);
    return (((z & M) << S3) ^ b);
}

uint LCGStep(uint z, uint A, uint C) {
    return (A * z + C);
}

float HybridTaus(uint seed) {
    // Combined Tausworthe generator
    seed = TausStep(seed, 13, 19, 12, 4294967294UL) ^ TausStep(seed, 2, 25, 4, 4294967288UL) ^
           TausStep(seed, 3, 11, 17, 4294967280UL) ^ LCGStep(seed, 1664525, 1013904223UL);

    // Convert to float
    return (seed & 0xFFFFFF) / 16777216.0f; // 2^24
}

// Generates a random unit vector in the hemisphere oriented around the given normal
float3 RandomUnitVectorInHemisphere(float3 normal, float random1, float random2) {
    // Convert uniform random numbers into spherical coordinates
    float phi = 2.0f * 3.14159265358979323846f * random1;
    float theta = acos(1.0 - random2);

    // Convert spherical coordinates to Cartesian coordinates on a unit sphere
    float x = sin(theta) * cos(phi);
    float y = sin(theta) * sin(phi);
    float z = cos(theta); // This ensures it's in the upper hemisphere

    float3 sphereDir = float3(x, y, z);

    // Create a coordinate system (tangent, bitangent, normal)
    float3 Nt, Nb;
    if (abs(normal.x) > abs(normal.z)) {
        Nt = float3(normal.y, -normal.x, 0.0);
    } else {
        Nt = float3(0.0, -normal.z, normal.y);
    }
    Nt = normalize(Nt);
    Nb = cross(normal, Nt);

    // Transform the sphere direction to align with the original normal
    float3 hemisphereDir = sphereDir.x * Nt + sphereDir.y * Nb + sphereDir.z * normal;
    return normalize(hemisphereDir);
}