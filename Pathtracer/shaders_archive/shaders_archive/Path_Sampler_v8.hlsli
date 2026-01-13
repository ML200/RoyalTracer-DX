// Define a path state object
struct PathState{
    float3 x; // current ray shading point
    float3 n; // Current ray normal; switch in case we transmit
    float3 o; // current outgoing direction
    uint objID; // object id of the mesh the shading point lies on
    uint matID; // material id of the mesh the shading point lies on
    uint ior_pointer = 0u; // What medium are we currently in?
    float ior_stack[8] = {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}; // stack of mediums for transmission
}

PathState initPathState(float3 x, float3 n, float3 o, uint objID, uint matID){
    PathState state;
    state.x = x;
    state.n = n;
    state.o = o;
    state.objID = objID;
    state.matID = matID;
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
}

// Sample a single backward bsdf ray based on the material properties
SampleState BSDF_BW_S(PathState pstate){

}