// RIS reservoir for direct lighting
struct Reservoir_DI
{
    float3 x2;    float w_sum; // 16 bytes total
    float3 n2;    float W;     // 16 bytes total
    float3 L2;     uint16_t M;     uint16_t  s; // 16 bytes total
};

struct Reservoir_GI
{
    float3 xn; // Reconnection vertex position (world space)
    float3 nn; // reconnection vertex normal
    float3 Vn; // Direction into reconnection vertex
    uint16_t  k; // Path index reconnection vertex
    float w_sum;
    float W;
    float3 f; // canonical contribution (before pdf)
    uint16_t M; // confidence weight
    uint16_t  s; //
    half3 E3;
    uint2 seed;
};

// Update the reservoir with the light
inline void UpdateReservoir_GI(
    inout Reservoir_GI reservoir,
    float wi,
    float M,

    float3 xn,
    float3 nn,
    float3 Vn,
    float3 E3,
    uint s,
    uint16_t k,
    float3 f,
    uint2 sample_seed,
    inout uint2 seed
    )
{

    reservoir.w_sum += wi;
    reservoir.M += (uint16_t)M;

    if (RandomFloat(seed) < wi / reservoir.w_sum)
    {
        reservoir.x2 = x2;
        reservoir.n2 = n2;
        reservoir.L2 = L2;
        reservoir.s = (uint16_t)s;
    }
}


// Update the reservoir with the light
inline void UpdateReservoir(
    inout Reservoir_DI reservoir,
    float wi,
    float M,

    float3 x2,
    float3 n2,
    float3 L2, // No need to update L1, as this is always 0 when the sample is processed here. Alwo,we dont want to reuse sample on a lights surface
    uint s,

    inout uint2 seed
    )
{

    reservoir.w_sum += wi;
    reservoir.M += (uint16_t)M;

    if (RandomFloat(seed) < wi / reservoir.w_sum)
    {
        reservoir.x2 = x2;
        reservoir.n2 = n2;
        reservoir.L2 = L2;
        reservoir.s = (uint16_t)s;
    }
}

inline void SetReservoirWeight(Reservoir_DI reservoir, float weight){
    reservoir.w_sum = weight;
}