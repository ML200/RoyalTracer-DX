// RIS reservoir for direct lighting
struct Reservoir_DI
{
    float3 x2;    float pad0; // 16 bytes total
    float3 n2;    float pad1; // 16 bytes total
    float w_sum;  float W;  float M;  float pad2; // 16 bytes total
    float3 L2;    float pad3; // 16 bytes total
    uint s;       uint pad4; uint pad5; uint pad6; // 16 bytes total (if needed)
};

struct Reservoir_GI
{
    float3 indirect; // For now
};

// Update the reservoir with the light
void UpdateReservoir(
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
    reservoir.M += M;

    if (RandomFloat(seed) < wi / reservoir.w_sum)
    {
        reservoir.x2 = x2;
        reservoir.n2 = n2;
        reservoir.L2 = L2;
        reservoir.s = s;
    }
}