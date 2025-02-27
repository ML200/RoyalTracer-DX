// RIS reservoir for direct lighting
struct Reservoir
{
    float3 x1;    // hit vertex
    float3 n1;    // hit normal
    float3 x2;    // Reconnection vertex position (object space)
    float3 n2;    // Reconnection normal (object space)
    float w_sum; // sum of weights
    float p_hat;    // target function (using in weight calulation etc)
    float W;     // Unbiased contribution weight
    float M;     // Number of candidates (c value)
    float V;     //visibility from postponed check
    float3 L1;    // If not 0, we hit a light and use this as pixel shading (outsource later)
    float3 L2;    // The selected lights emission
    uint s;     // Strategy for lobe sampling
    float3 o;     // Outgoing ray (outsource later)
    uint mID;    // Material ID (outsource later)
};


// Update the reservoir with the light
void UpdateReservoir(
    inout Reservoir reservoir,
    float wi,
    float M,
    float p_hat,

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
        reservoir.p_hat = p_hat;
        reservoir.L2 = L2;
        reservoir.s = s;
    }
}