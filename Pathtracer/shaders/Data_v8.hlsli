struct STriVertex {
    float3 vertex;
    uint   packedNormal;
    half2  texCoord;
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
    uint opaqueTriCount;
    uint _pad[3];
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
    float3 Tf;

    float2 albedoUVScale;
    float2 normalUVScale;
    float2 rmaUVScale;

    int albedoTexID;
    int normalTexID;
    int rmaTexID;
    float alphaThreshold;
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