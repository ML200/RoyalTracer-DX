/*
Class for sampling path segements
*/

// Define a path state object
struct PathState{
    float3 x; // current ray shading point
    float3 n_g; // geometric normal
    float3 n_s; // shading normal
    float3 o; // current outgoing direction
    uint objID; // object id of the mesh the shading point lies on
    uint matID; // material id of the mesh the shading point lies on
    uint ior_pointer; // What medium are we currently in?
    float ior_stack[4]; // stack of mediums for transmission
    float priority_stack[4]; // stack priority of objects we currently traverse
};

PathState InitPathState(float3 x, float3 n_g, float3 n_s, float3 o, uint objID, uint matID){
    PathState pstate;
    pstate.x = x;
    pstate.n_g = n_g;
    pstate.n_s = n_s;
    pstate.o = o;
    pstate.objID = objID;
    pstate.matID = matID;

    [unroll] // Stupid hlsl
    for (int i = 0; i < 4; ++i) pstate.ior_stack[i] = 0.0f;
    for (int i = 0; i < 4; ++i) pstate.priority_stack[i] = 0.0f;
    return pstate;
}

// Storage for the current state of the path up until this path vertex
struct ThroughputState{
    float3 t;
    float pdf;
};

ThroughputState InitThroughputState(){
    ThroughputState tstate;
    tstate.t = float3(1.0f, 1.0f, 1.0f);
    tstate.pdf = 1.0f;
    return tstate;
}

// Samplers output a SampleState object that contains information about the surface hit etc
// Different to PathState objects, they shouldnt be persistent and just be used as containers for data in sampler calls
struct SampleState{
    float3 x;
    float3 s; // The sample direction
    float3 n_g; // Geometric vs shading normal of the hit surface
    float3 n_s;
    float3 o;
    float3 L; // Theoretical emission
    uint matID;
    uint objID;
    uint lightID; // Did we hit an emitter? If not the id is 0xFFFFFFFF
    bool b; // Did we hit a backface?
};

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
    float3 s = SampleBRDF(pstate.matID, pstate.o, pstate.n_g, pstate.n_s, rdata);
    if(all(s == 0.0f))
        return (SampleState)0; // Invalid sample: geometric normal is 0

    // Trace the ray
    RayDesc ray;
    ray.Origin = pstate.x;
    ray.Direction = s;
    ray.TMin = 0.00001f;
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