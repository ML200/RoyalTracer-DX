struct STriVertex {
    float3 vertex;
    float4 normal;
    float2 texCoord;
};

cbuffer Colors : register(b0) {
    float3 A;
    float3 B;
    float3 C;
}

struct Attributes {
    float2 bary;
};

struct InstanceProperties
{
    float4x4 objectToWorld;
    float4x4 objectToWorldInverse;
    float4x4 prevObjectToWorld;
    float4x4 prevObjectToWorldInverse;
    float4x4 objectToWorldNormal;
    float4x4 prevObjectToWorldNormal;
    uint  indexBase;
    uint  vertexBase;
    uint  materialBase;
    uint triToLightBase;
};

struct LightTriangle {
    float3 x;
    float cdf;       // 16 bytes
    float3 y;
    uint instanceID; // 16 bytes
    float3 z;
    float weight;       // 16 bytes
    float3 emission;
    uint triCount;   // 16 bytes
    float total_weight;
    float3 pad0;       // 16 bytes
};

struct Material
{
    float4 Kd;
    float3 Ke;
    float Ni;
    float4 Pr_Pm_Ps_Pc;
    float3 Pcr_aniso_anisor;
    float pcr_pad;
    float LUT[16];
    float SheenLUT[16];
    int albedoTexID;
    int normalTexID;
    int rmaTexID;
    int pad;
};

// Ray payloads for closest hit and shadow ray
struct [[raypayload]] HitInfo {
    float3 hitPosition : read(caller)
                         : write(anyhit,closesthit,miss);
    uint materialID : read(caller)
                         : write(anyhit,closesthit,miss);
    float3 hitNormal : read(caller)
                         : write(anyhit,closesthit,miss);
    float3 hitGNormal : read(caller)
                         : write(anyhit,closesthit,miss);
    float area: read(caller)
                         : write(anyhit,closesthit,miss);
    uint objID: read(caller)
                         : write(anyhit,closesthit,miss);
    uint lightID: read(caller)
                         : write(anyhit,closesthit,miss);
    bool hitBackface: read(caller)
                         : write(anyhit,closesthit,miss);
};

struct [[raypayload]] ShadowHitInfo {
    bool isHit: read(caller)
                         : write(anyhit,closesthit,miss);
};

struct SampleReturn
{
    float3 x2;
    float3 n2;
    float3 L2;
    uint objID; // Of the light/x2 hit
    uint matID; // Of the light/x2 hit
    float pdf_bsdf;
    float pdf_nee;
};

// Extended sample return for bdpt
struct BDReturn
{
    float3 x2;
    float3 n2;
    float3 L2;
    uint objID; // Of the light/x2 hit
    uint matID; // Of the light/x2 hit
    float pdf;
    float pdf_seg;

    float3 x3; // third vertex reconnection data
    float3 n3;
    uint triID;
};


struct WeightedPixel
{
    int2  pix;
    float w;
};


struct LightTLASNodeGpu
{
    float3 bmin;     float power;
    float3 bmax;     float theta_o;
    float3 axis;     float theta_e;

    uint   firstChild;
    uint   childCount;
    uint   blasIndex;
    uint   primCount;

    float  sumPower;
    float  sumPowerSq;

    uint   itemFirst;
    uint   itemCount;
};

struct LightBLASNodeGpu
{
    float3 bmin;     float power;
    float3 bmax;     float theta_o;
    float3 axis;     float theta_e;

    uint   firstChild;
    uint   childCount;
    uint   triFirst;
    uint   triCount;

    uint   primCount;     uint _pad0;
    float  sumPower;      float sumPowerSq;
};

struct BlasRangeGpu {
    uint nodeOffset;
    uint nodeCount;
    uint triIndexOffset;
    uint triIndexCount;
};


struct LT_Sample { uint id; float pdf; };
struct LT_Path_Sample { float3 dir; float pdf; uint tri;};


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

// Storage for the current state of the path up until this path vertex
struct ThroughputState{
    float3 t;
    float pdf;
};

// Define a path state object
struct PathState{
    float3 x; // current ray shading point
    float3 n_g; // geometric normal
    float3 n_s; // shading normal
    float3 o; // current outgoing direction
    uint objID; // object id of the mesh the shading point lies on
    uint matID; // material id of the mesh the shading point lies on
    int ior_pointer; // What medium are we currently in?
    float ior_stack[4]; // stack of mediums for transmission
    float priority_stack[4]; // stack priority of objects we currently traverse
};