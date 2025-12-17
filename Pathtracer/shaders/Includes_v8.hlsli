cbuffer Push : register(b1)
{
    uint2 gImageSize;       // [0,1] Width, Height (Legacy name)
    uint  g_InputStackIdx;  // [2]   Current Stack to read from
    uint  g_OutputStackIdx; // [3]   Next Stack to write to
    uint2 _Padding;         // [4,5] Unused (Alignment)
};

#define ENABLE_RAY_QUERY_INLINE // Activate support for inline ray tracing

#define gImageWidth  (gImageSize.x)
#define gImageHeight (gImageSize.y)
#define IMG_W        (gImageSize.x)
#define IMG_H        (gImageSize.y)

#define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)

static uint3 gDispatchIdx;
#define DispatchRaysIndex()      gDispatchIdx

SamplerState g_sampler : register(s0);

Texture2DArray albedoTextures : register(t30);
Texture2DArray normalTextures : register(t31);
Texture2DArray rmaTextures    : register(t32);

SamplerState g_sampler_LUT : register(s1);

Texture2DArray g_LUT : register(t33);

// Wavefront compaction stuff
RWByteAddressBuffer g_GlobalCounters : register(u34);
RWStructuredBuffer<uint3> g_IndirectArgs : register(u35);
RWStructuredBuffer<uint2> g_Stack0 : register(u36);
RWStructuredBuffer<uint2> g_Stack1 : register(u37);

RWByteAddressBuffer g_SortCount  : register(u60);
RWByteAddressBuffer g_SortOffset : register(u61);
RWByteAddressBuffer g_SortBounds : register(u62);


#include "Constants_v8.hlsli"
#include "Common_v8.hlsli"
#include "Data_v8.hlsli"
#include "Random_v8.hlsli"
#include "Compression_v8.hlsli"
#include "HitState_v8.hlsli"

RWTexture2DArray<float4> gOutput             : register(u0);
RWTexture2D<float4>      gPermanentData      : register(u1);
RWTexture2DArray<float4> gScratchPing         : register(u8);

RWByteAddressBuffer g_sample_current         : register(u6);
RWByteAddressBuffer g_sample_last            : register(u7);
RWByteAddressBuffer g_Reservoirs_current_di  : register(u2);
RWByteAddressBuffer g_Reservoirs_last_di     : register(u3);
RWByteAddressBuffer g_Reservoirs_current_gi  : register(u4);
RWByteAddressBuffer g_Reservoirs_last_gi     : register(u5);
RWByteAddressBuffer g_InitialBSDFRays        : register(u9);
RWByteAddressBuffer g_pathStateBuffer : register(u10);

StructuredBuffer<STriVertex>          BTriVertex        : register(t2);
StructuredBuffer<int>                 indices           : register(t1);
RaytracingAccelerationStructure       SceneBVH          : register(t0);
StructuredBuffer<InstanceProperties>  instanceProps     : register(t3);
StructuredBuffer<uint>                materialIDs       : register(t4);
StructuredBuffer<Material>            materials         : register(t5);
StructuredBuffer<LightTriangle>       g_EmissiveTriangles : register(t6);
StructuredBuffer<float>               g_AliasProb       : register(t7);
StructuredBuffer<uint>                g_AliasIdx        : register(t8);

StructuredBuffer<uint> gTriToLightId     : register(t15);

// Light tree
StructuredBuffer<LightTLASNodeGpu> gLT_TLAS        : register(t9);
StructuredBuffer<LightBLASNodeGpu> gLT_BLAS        : register(t10);
StructuredBuffer<BlasRangeGpu>     gLT_Range       : register(t11);
Buffer<uint>                       gLT_LeafTriIndex: register(t12);
Buffer<float>          gLT_LeafAliasProb : register(t16);
Buffer<uint>           gLT_LeafAliasIdx  : register(t17);


// Needs access to all structured/random buffers
#include "LightTree_v8.hlsli"
#include "Sample_Data_v8.hlsli"
#include "Path_State_v8.hlsli"
#include "Fresnel_v8.hlsli"
#include "Material_Common_v8.hlsli"
#include "Material_GGX_v8.hlsli"
#include "Material_Lambertian_v8.hlsli"
#include "Material_Coat_v8.hlsli"
#include "Material_Sheen_v8.hlsli"
#include "BXDF_v8.hlsli"

cbuffer CameraParams : register(b0)
{
    float4x4 view;
    float4x4 projection;
    float4x4 viewI;
    float4x4 projectionI;
    float4x4 prevView;
    float4x4 prevProjection;
    float time;
}
// These includes need access to ALL previous buffers
#include "Reservoir_DI_v8.hlsli"
#include "Reservoir_GI_v8.hlsli"
#include "Inline_RT_v8.hlsli"
#include "Camera_ray_v8.hlsli"
#include "Path_Sampler_v8.hlsli"