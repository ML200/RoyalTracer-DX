struct STriVertex {
    float3 vertex;
    uint   packedNormal;
    half2  texCoord;
};

struct Attributes {
    float2 bary;
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

// Full-precision storage of the SG light clusters (matches lt::LightTLASNodeGpu and
// lt::LightBLASNodeGpu on the host); the compact layouts live in LightTreePacked.h.
struct LightTLASNodeFull
{
    float3 mean;     float variance;
    float3 rbar;     float power;
    float  radius;   float cosTheta_o; uint firstChild; uint childCount;
    uint   slot;     uint3 _pad;
};
struct LightBLASNodeFull
{
    float3 mean;     float variance;
    float3 rbar;     float power;
    float  radius;   float cosTheta_o; uint firstChild; uint childCount;
    uint   triFirst; uint  triCount;   uint2 _pad;
};

// Traversal values; the selected GPU layout is decoded by LightTreeDecode.hlsli. The mean
// resultant vector of the emission directions becomes the vMF axis and sharpness.
struct LightTLASNodeGpu
{
    float3 mean;     float variance;
    float3 axis;     float kappa;
    float  power;    float radius;   float cosTheta_o;

    uint   firstChild;
    uint   childCount;
    uint   slot;
};

struct LightBLASNodeGpu
{
    float3 mean;     float variance;
    float3 axis;     float kappa;
    float  power;    float radius;   float cosTheta_o;

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
struct LT_Path_Sample { float3 dir; float pdf; uint tri;};
