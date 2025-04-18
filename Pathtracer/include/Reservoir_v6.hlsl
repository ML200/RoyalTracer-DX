// Each row is 16 bytes
struct SampleData
{
    float3 x1;  // (12 bytes)
    uint16_t  mID;   // (2 bytes)
    half3 L1;   // (6 bytes)
    float3 n1;  // (12 bytes)
    float3 o;    // (12 bytes)
    uint  objID;   // (4 bytes)
    float3  debug;   // (12 bytes)
};


// RIS reservoir for direct lighting
struct Reservoir_DI
{
    float3 x2;    float w_sum; // 16 bytes total
    float3 n2;    float W;     // 16 bytes total
    half3 L2;     uint16_t M;   // 8 bytes total
};

struct Reservoir_GI
{
    float3 xn;     float w_sum; //16
    float3 nn;     float W;  //16
    half3 E3;     uint16_t M; //8
};

// Update the reservoir with the light
inline bool UpdateReservoir_GI(
    inout Reservoir_GI reservoir,
    float wi,
    float M,

    float3 xn,
    float3 nn,
    float3 E3,
    inout uint2 seed
    )
{

    reservoir.w_sum += wi;
    reservoir.M += M;

    if (RandomFloat(seed) < wi / reservoir.w_sum)
    {
        reservoir.xn = xn;
        reservoir.nn = nn;
        reservoir.E3 = E3;
        return true;
    }
    return false;
}


// Update the reservoir with the light
inline bool UpdateReservoir(
    inout Reservoir_DI reservoir,
    float wi,
    float M,

    float3 x2,
    float3 n2,
    float3 L2, // No need to update L1, as this is always 0 when the sample is processed here. Alwo,we dont want to reuse sample on a lights surface
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
        return true;
    }
    return false;
}

inline void SetReservoirWeight(Reservoir_DI reservoir, float weight){
    reservoir.w_sum = weight;
}

inline void SetReservoirWeight_GI(Reservoir_DI reservoir, float weight){
    reservoir.w_sum = weight;
}