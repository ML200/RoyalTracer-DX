//====================================
//VERTEX AND ATTRIBUTES
//====================================
struct STriVertex {
    float3 vertex;
    uint   packedNormal;
    half2  texCoord;
};

struct Attributes {
    float2 bary;
};

//====================================
//INSTANCE PROPERTIES
//====================================
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
    uint opaqueTriCount;
    uint _pad[3];
};

//====================================
//LIGHT TRIANGLE
//====================================
struct LightTriangle {
    float3 x;
    float cdf;
    float3 y;
    uint instanceID;
    float3 z;
    float weight;
    float3 emission;
    uint triCount;
    float total_weight;
    float3 pad0;
};

//====================================
//PACKED MATERIAL
//====================================
//40 B AoS, one material per cache line, path threads hit random materials
//Ke dropped on CPU, emission via LightTriangle buffer
//offset 0  Kd_rgb                RGB9E5
//offset 4  w_Ni                  half Kd.w | half Ni
//offset 8  PrPmPsPc              4x uint8 [0,1], LSB=Pr
//offset 12 Tf_rgb                RGB9E5
//offset 16 Pcr_Aniso_Rot_AlphaTh u8 Pcr | i8 aniso | u8 anisoRot | u8 alphaTh
//offset 20 texIDs_01             i16 albedo | i16 normal
//offset 24 texIDs_2              i16 rma    | i16 pad
//offset 28 uv_albedo             half2
//offset 32 uv_normal             half2
//offset 36 uv_rma                half2
struct MatPacked {
    uint Kd_rgb;
    uint w_Ni;
    uint PrPmPsPc;
    uint Tf_rgb;
    uint Pcr_Aniso_Rot_AlphaTh;
    uint texIDs_01;
    uint texIDs_2;
    uint uv_albedo;
    uint uv_normal;
    uint uv_rma;
};

//====================================
//LIGHT TREE NODES
//====================================
struct LightTLASNodeGpu
{
    float3 bmin;     float power;
    float3 bmax;     float cosTheta_o;
    float3 axis;     float cosTheta_e;

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
    float3 bmax;     float cosTheta_o;
    float3 axis;     float cosTheta_e;

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
    float4x4 worldToLocal;
};


struct LT_Sample { uint id; float pdf; };
struct LT_Path_Sample { float3 dir; float pdf; uint tri;};
