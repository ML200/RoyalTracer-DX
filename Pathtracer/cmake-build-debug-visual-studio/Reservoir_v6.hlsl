// RIS reservoir for direct lighting
struct Reservoir
{
    // Current sample parameters
    float3 f;
    float p_hat;
    bool v_eval; // visibility for this sample already evaluated?

    float w_sum; // sum of weights
    float M; // Number of candidates
};


// Update the reservoir with the light
bool UpdateReservoir(
    inout Reservoir reservoir,
    float wi,
    inout uint2 seed,

    float3 f,
    float p_hat,
    bool v_eval
    )
{
    reservoir.w_sum += wi;
    reservoir.M += 1.0f;

    if ( RandomFloat(seed) < wi / reservoir.w_sum  )
    {
        reservoir.f = f;
        reservoir.p_hat = p_hat;
        reservoir.v_eval = v_eval;
        return true;
    }

    return false;
}










