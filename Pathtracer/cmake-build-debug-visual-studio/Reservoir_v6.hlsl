// RIS reservoir for direct lighting
struct Reservoir
{
    // Current sample parameters
    float3 f;
    float p_hat;
    float3 direction;
    float dist;
    float3 hitPos;
    float3 hitNormal;
    bool v_eval; // visibility for this sample already evaluated?
    float v;

    float w_sum; // sum of weights
    float w_i;
    float M; // Number of candidates
};


// Update the reservoir with the light
bool UpdateReservoir(
    inout Reservoir reservoir,
    float wi,
    float M,
    inout uint2 seed,

    float3 f,
    float p_hat,
    bool v_eval,
    float v,
    float3 direction,
    float dist,
    float3 hitPos,
    float3 hitNormal
    )
{

    reservoir.w_sum += wi;
    reservoir.M += M;

    if (RandomFloat(seed) < wi / max(EPSILON, reservoir.w_sum))
    {
        reservoir.f = f;
        reservoir.p_hat = p_hat;
        reservoir.v_eval = v_eval;
        reservoir.direction =direction;
        reservoir.dist = dist;
        reservoir.hitPos = hitPos;
        reservoir.hitNormal = hitNormal;
        reservoir.v = v;
        if(reservoir.p_hat > 0.0f)
            reservoir.w_i = reservoir.w_sum / p_hat;
        return true;
    }
    if(reservoir.p_hat > 0.0f)
        reservoir.w_i = reservoir.w_sum / reservoir.p_hat;
    else
        reservoir.w_i = 0.0f;
    return false;
}



// Weight the reservoir.
void WeightReservoir(
    inout Reservoir reservoir,
    float w
    )
{
    reservoir.w_sum *= w;
}
