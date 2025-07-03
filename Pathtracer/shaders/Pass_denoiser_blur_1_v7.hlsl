cbuffer Push : register(b1) { uint2 gImageSize; }
#define gImageWidth   (gImageSize.x)
#define gImageHeight  (gImageSize.y)
#define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)

groupshared uint3 gDispatchIdxShared[1];
static     uint3  gDispatchIdx;
#define DispatchRaysIndex() gDispatchIdx

#include "Constants_v7.hlsli"
#include "Common_v7.hlsli"
#include "Structures_misc.hlsli"
#include "Random_v7.hlsli"
#include "Compression_v7.hlsli"

RWTexture2DArray<half4> gOutput              : register(u0);
RWTexture2D<half4>      gPermanentData       : register(u1);
RWTexture2DArray<half4>      gScratchPing         : register(u8);

RWByteAddressBuffer      g_sample_current         : register(u6);
RWByteAddressBuffer      g_sample_last            : register(u7);
RWByteAddressBuffer      g_Reservoirs_current_di  : register(u2);
RWByteAddressBuffer      g_Reservoirs_last_di     : register(u3);
RWByteAddressBuffer      g_Reservoirs_current_gi  : register(u4);
RWByteAddressBuffer      g_Reservoirs_last_gi     : register(u5);

StructuredBuffer<STriVertex>  BTriVertex           : register(t2);
StructuredBuffer<int>         indices              : register(t1);
RaytracingAccelerationStructure SceneBVH           : register(t0);
StructuredBuffer<InstanceProperties> instanceProps : register(t3);
StructuredBuffer<uint>             materialIDs     : register(t4);
StructuredBuffer<Material>         materials       : register(t5);
StructuredBuffer<LightTriangle>    g_EmissiveTriangles : register(t6);
StructuredBuffer<float>            g_AliasProb     : register(t7);
StructuredBuffer<uint>             g_AliasIdx      : register(t8);

#include "Sample_data.hlsli"
#include "GGX_v7.hlsli"
#include "Lambertian_v7.hlsli"
#include "BSDF_v7.hlsli"

cbuffer CameraParams : register(b0)
{
    float4x4 view; float4x4 projection; float4x4 viewI; float4x4 projectionI;
    float4x4 prevView; float4x4 prevProjection; float time;
}

#include "Camera_ray_v7.hlsli"
#include "NEE_Sampling_v7.hlsli"
#include "Reservoir_DI_v7.hlsli"
#include "Motion_vectors_v7.hlsli"
#include "Denoiser_helper_v7.hlsli"

[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    gDispatchIdx = DTid;
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;

    uint2  launch = DTid.xy;
    uint2  dimsI  = DispatchRaysDimensions().xy;
    float2 dims   = float2(dimsI);
    uint   pIdx   = MapPixelID(dims, launch);

    // Blur kernel
    float3 output = AtrousKernel(launch, 4, 0);

    // Store accumulated result
    //gPermanentData[DTid.xy] = float4(output, 1);
    gScratchPing[uint3(launch, 1)] = float4(output, 1);
}


