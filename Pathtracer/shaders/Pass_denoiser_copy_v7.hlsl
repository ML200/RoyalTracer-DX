// ─────────────────────────────────────────────────────────────────────────────
//  ROOT-CONSTANTS  (slot 1 : two uints = 8 bytes)
//
//  These map 1-to-1 to the `Set*Root32BitConstants` call in Renderer.cpp.
//──────────────────────────────────────────────────────────────────────────────
cbuffer Push : register(b1)
{
    uint2 gImageSize;
}

// Convenience aliases – some of the legacy headers still expect them
#define gImageWidth   (gImageSize.x)
#define gImageHeight  (gImageSize.y)

// ─────────────────────────────────────────────────────────────────────────────
//  Fake the two DXR intrinsics that many helpers rely on.
//  We just substitute the thread-ID and the constant dimensions.
//──────────────────────────────────────────────────────────────────────────────
#define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)

// DTid is only visible inside `main`, so we stash a copy in a globals so that
// the macro below can see it.  Each thread overwrites its own instance, so no
// synchronisation is needed.
static uint3 gDispatchIdx;
#define DispatchRaysIndex()      gDispatchIdx


#include "Constants_v7.hlsli"
#include "Common_v7.hlsli"
#include "Structures_misc.hlsli"
#include "Random_v7.hlsli"
#include "Compression_v7.hlsli"

RWTexture2DArray<float4> gOutput : register(u0);
RWTexture2D<float4> gPermanentData : register(u1);
RWTexture2D<float4> gScratchPing : register(u8); // Storage for denoiser

RWByteAddressBuffer g_sample_current : register(u6);
RWByteAddressBuffer g_sample_last : register(u7);
RWByteAddressBuffer g_Reservoirs_current_di : register(u2);
RWByteAddressBuffer g_Reservoirs_last_di : register(u3);
RWByteAddressBuffer g_Reservoirs_current_gi : register(u4);
RWByteAddressBuffer g_Reservoirs_last_gi : register(u5);

StructuredBuffer<STriVertex> BTriVertex : register(t2);
StructuredBuffer<int> indices : register(t1);
RaytracingAccelerationStructure SceneBVH : register(t0);
StructuredBuffer<InstanceProperties> instanceProps : register(t3);
StructuredBuffer<uint> materialIDs : register(t4);
StructuredBuffer<Material> materials : register(t5);
StructuredBuffer<LightTriangle> g_EmissiveTriangles : register(t6);
StructuredBuffer<float> g_AliasProb  : register(t7);
StructuredBuffer<uint>  g_AliasIdx   : register(t8);

// Needs access to all structured/random buffers
#include "Sample_data.hlsli"
#include "GGX_v7.hlsli"
#include "Lambertian_v7.hlsli"
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
#include "Camera_ray_v7.hlsli"
#include "NEE_Sampling_v7.hlsli"
#include "Reservoir_DI_v7.hlsli"
#include "Motion_vectors_v7.hlsli"

//──────────────────────────────────────────────────────────────────────────────
//  Compute kernel: 8×8 threads
//──────────────────────────────────────────────────────────────────────────────
[numthreads(16, 16, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    // Guard against fringe threads in a non-multiple dispatch.
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;


    // Make the faux-intrinsics macros work for any helper that uses them.
    gDispatchIdx = uint3(DTid.xy, 0);

    // Pixel linear index
    uint  pixelIdx = MapPixelID(float2(gImageWidth, gImageHeight), DTid.xy);

    // Load current-frame data and push it into the “last-frame” buffer.
    SampleData cur = loadSampleData(g_sample_current, pixelIdx);

    store_x1   (cur.x1   , g_sample_last, pixelIdx);
    store_n1   (cur.n1   , g_sample_last, pixelIdx);
    store_L1   (cur.L1   , g_sample_last, pixelIdx);
    store_o    (cur.o    , g_sample_last, pixelIdx);
    store_matID(cur.matID, g_sample_last, pixelIdx);
    store_objID(cur.objID, g_sample_last, pixelIdx);
}
