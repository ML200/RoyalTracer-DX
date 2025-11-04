/*
Class for sampling path segements
*/

// Define a path state object
struct PathState{
    float3 x; // current ray shading point
    float3 n; // Current ray normal; switch in case we transmit
    float3 o; // current outgoing direction
    uint objID; // object id of the mesh the shading point lies on
    uint matID; // material id of the mesh the shading point lies on
    uint ior_pointer; // What medium are we currently in?
    float ior_stack[4]; // stack of mediums for transmission
    float priority_stack[4]; // stack priority of objects we currently traverse
};

PathState initPathState(float3 x, float3 n, float3 o, uint objID, uint matID){
    PathState pstate;
    pstate.x = x;
    pstate.n = n;
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

ThroughputState initThroughputState(){
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
    float3 n;
    float3 o;
    uint matID;
    uint objID;
    bool b; // Did we hit a backface?
    bool l; // Did we hit an emitter?
};

// Sample a single backward bsdf ray based on the material properties
/*SampleState BSDF_BW_S(PathState pstate){

}*/