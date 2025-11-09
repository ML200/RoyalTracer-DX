/*
Class for sampling path segements
*/

PathState InitPathState(float3 x, float3 n_g, float3 n_s, float3 o, uint objID, uint matID){
    PathState pstate = (PathState)0.0f;
    pstate.x = x;
    pstate.n_g = n_g;
    pstate.n_s = n_s;
    pstate.o = o;
    pstate.objID = objID;
    pstate.matID = matID;

    pstate.ior_pointer = -1; // We start in the air.

    [unroll] // Stupid hlsl
    for (int i = 0; i < 4; ++i) pstate.ior_stack[i] = 0.0f;
    for (int i = 0; i < 4; ++i) pstate.priority_stack[i] = 0.0f;
    return pstate;
}

ThroughputState InitThroughputState(){
    ThroughputState tstate;
    tstate.t = float3(1.0f, 1.0f, 1.0f);
    tstate.pdf = 1.0f;
    return tstate;
}

// Update the path state with a sample
void AdvancePathState(SampleState sstate, inout PathState pstate){
    pstate.x = sstate.x;
    pstate.n_g = sstate.n_g;
    pstate.n_s = sstate.n_s;
    pstate.o = sstate.o;
    pstate.objID = sstate.objID;
    pstate.matID = sstate.matID;
}

// Helper to check if we have a termination condition
float3 Get_Emissive(HitInfo h){
    if(h.materialID == 0xFFFFFFFF) // we hit the sky
        return h.hitPosition;
    if(h.hitBackface)
        return float3(0,0,0);
    return materials[h.materialID].Ke.xyz;
}

// Helper to check if a given sample state is valid
bool ValidSampleState(SampleState sstate){
    if(length(sstate.n_g) < EPSILON) return false;
    return true;
}

// Sample a single backward bsdf ray based on the material properties
SampleState Sample_BSDF_BW_S(PathState pstate, inout RandomData rdata){
    // Sample a BSDF direction
    float3 s = SampleBRDF(pstate, rdata);
    if(all(s == 0.0f))
        return (SampleState)0; // Invalid sample: geometric normal is 0

    // Trace the ray
    RayDesc ray;
    ray.Origin = pstate.x;
    ray.Direction = s;
    ray.TMin = 0.0001f;
    ray.TMax = 10000.0f;
    HitInfo payload = (HitInfo)0.0f;
    TraceRayInline_HitInfo(SceneBVH, ray, payload, RAY_FLAG_NONE, 0xFF);

    SampleState sstate;
    sstate.x = payload.hitPosition;
    sstate.s = s;
    sstate.n_g = payload.hitGNormal;
    sstate.n_s = payload.hitNormal;
    sstate.o = normalize(pstate.x - payload.hitPosition);
    sstate.L = Get_Emissive(payload);
    sstate.matID = payload.materialID;
    sstate.objID = payload.objID;
    sstate.b = payload.hitBackface;
    sstate.lightID = sstate.b ? 0xFFFFFFFF : payload.lightID; // If we hit a backface, it doesnt count as light
    return sstate;
}