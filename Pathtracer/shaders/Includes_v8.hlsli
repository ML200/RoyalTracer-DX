cbuffer Push : register(b1)
{
    uint2 gImageSize;
}
#define ENABLE_RAY_QUERY_INLINE // Activate support for inline ray tracing

#define gImageWidth  (gImageSize.x)
#define gImageHeight (gImageSize.y)
#define IMG_W        (gImageSize.x)
#define IMG_H        (gImageSize.y)

#define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)

static uint3 gDispatchIdx;
#define DispatchRaysIndex()      gDispatchIdx

#include "Constants_v8.hlsli"
#include "Common_v8.hlsli"
#include "Structures_v7.hlsli"
#include "Random_v7.hlsli"
#include "Compression_v7.hlsli"

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

StructuredBuffer<STriVertex>          BTriVertex        : register(t2);
StructuredBuffer<int>                 indices           : register(t1);
RaytracingAccelerationStructure       SceneBVH          : register(t0);
StructuredBuffer<InstanceProperties>  instanceProps     : register(t3);
StructuredBuffer<uint>                materialIDs       : register(t4);
StructuredBuffer<Material>            materials         : register(t5);
StructuredBuffer<LightTriangle>       g_EmissiveTriangles : register(t6);
StructuredBuffer<float>               g_AliasProb       : register(t7);
StructuredBuffer<uint>                g_AliasIdx        : register(t8);

// Light tree
StructuredBuffer<LightTLASNodeGpu> gLT_TLAS        : register(t9);
StructuredBuffer<LightBLASNodeGpu> gLT_BLAS        : register(t10);
StructuredBuffer<BlasRangeGpu>     gLT_Range       : register(t11);
Buffer<uint>                       gLT_LeafTriIndex: register(t12);
Buffer<float>                      gLT_LeafAliasProb : register(t13);
Buffer<uint>                       gLT_LeafAliasIdx  : register(t14);


// Needs access to all structured/random buffers
#include "LightTree_v7.hlsli"
#include "Sample_data_v7.hlsli"
#include "Initial_bsdf_v7.hlsli"
#include "GGX_v7.hlsli"
#include "Lambertian_v7.hlsli"
#include "Coat_v7.hlsli"
#include "Sheen_v7.hlsli"
#include "GGXT_v7.hlsli"
#include "BSDF_v7.hlsli"

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
#include "Reservoir_DI_v7.hlsli"
#include "Reservoir_GI_v7.hlsli"
#include "Inline_RT_v7.hlsli"
#include "Camera_ray_v7.hlsli"
#include "MIS_v7.hlsli"
#include "NEE_Sampling_v7.hlsli"
#include "BSDF_Sampling_v7.hlsli"
#include "BD_Sampling_v7.hlsli"
#include "BD_Reconnection_v7.hlsli"
#include "Path_Sampler_v8.hlsli"
#include "Motion_vectors_v7.hlsli"
#include "Denoiser_helper_v7.hlsli"