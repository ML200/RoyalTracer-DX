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
  float util; //IMPORTANT: util info: miss flag
  uint seed;
  float pdf;
};

// Attributes output by the raytracing when hitting a surface,
// here the barycentric coordinates
struct Attributes {
  float2 bary;
};

// Hash function to scramble the bits of an integer
uint hash(uint x) {
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

// Returns a float in [0, 1)
float RandomFloat(uint seed) {
    const uint prime1 = 0x68bc21ebu; // Large prime number
    const uint prime2 = 0x02e5be93u; // Another large prime number

    uint scrambled = hash(seed * prime1 + prime2);
    return float(scrambled) * (1.0 / 4294967296.0); // 2^-32
}

// Main function to generate a random unit vector in the hemisphere
float3 RandomUnitVectorInHemisphere(float3 normal, inout uint seed) {
    // Generate two random numbers
    float random1 = RandomFloat(seed);
    float random2 = RandomFloat(seed);

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