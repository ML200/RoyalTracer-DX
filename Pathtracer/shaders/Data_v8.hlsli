struct STriVertex {
    float3 vertex;
    uint   packedNormal;
    half2  texCoord;
};

// Must match the host-side instance layout.
struct InstanceProperties
{
    float3x4 objectToWorld;
    float3x4 objectToWorldInverse;
    float3x4 objectToWorldNormal;
    uint  indexBase;
    uint  vertexBase;
    uint  materialBase;
    uint  triToLightBase;
    uint  opaqueTriCount;
    uint  _pad[2];
    uint  lightSlot;
    float3x4 prevObjectToWorld;
};

struct LightTriangle {
    float3 x;
    float cdf;
    float3 y;
    uint meshID;
    float3 z;
    float weight;
    float3 emission;
    uint triCount;
    float total_weight;
    float3 pad0;
};

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
    uint sss_albedo;
    uint sss_radius_g;
};

#include "LightTreePacked.h"

// Decoded form; see LightTreeDecode.hlsli.
struct LightTLASNodeGpu
{
    float3 bmin;     float power;
    float3 bmax;     float cosTheta_o;
    float3 axis;     float sinTheta_o;

    uint   firstChild;
    uint   childCount;
    uint   slot;
    uint   _pad;
};

struct LightBLASNodeGpu
{
    float3 bmin;     float power;
    float3 bmax;     float cosTheta_o;
    float3 axis;     float sinTheta_o;

    uint   firstChild;
    uint   childCount;
    uint   triFirst;
    uint   triCount;
};

struct BlasRangeGpu {
    uint nodeOffset;
    uint nodeCount;
    uint triIndexOffset;
    uint triIndexCount;
};

struct LightSlotGpu {
    float3x4 worldToLocal;
    uint  instanceID;
    uint  nodeOffset;
    float powerScale;
    uint  _pad;
};

struct LT_Sample { uint id; uint inst; float pdf; uint2 learningToken; };
