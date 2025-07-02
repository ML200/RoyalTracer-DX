cbuffer Push : register(b1)
{
    uint2 gImageSize;
}

#define gImageWidth   (gImageSize.x)
#define gImageHeight  (gImageSize.y)

#define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)

static uint3 gDispatchIdx;
#define DispatchRaysIndex()      gDispatchIdx

#include "Constants_v7.hlsli"
#include "Common_v7.hlsli"
#include "Structures_misc.hlsli"
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

StructuredBuffer<STriVertex>          BTriVertex        : register(t2);
StructuredBuffer<int>                 indices           : register(t1);
RaytracingAccelerationStructure       SceneBVH          : register(t0);
StructuredBuffer<InstanceProperties>  instanceProps     : register(t3);
StructuredBuffer<uint>                materialIDs       : register(t4);
StructuredBuffer<Material>            materials         : register(t5);
StructuredBuffer<LightTriangle>       g_EmissiveTriangles : register(t6);
StructuredBuffer<float>               g_AliasProb       : register(t7);
StructuredBuffer<uint>                g_AliasIdx        : register(t8);

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

// ─────────────────────────────────────────────────────────────────────────────
//  SHADING PASS
// ─────────────────────────────────────────────────────────────────────────────
[numthreads(8, 4, 1)]
void main(uint3 DTid : SV_DispatchThreadID)
{
    if (DTid.x >= gImageWidth || DTid.y >= gImageHeight) return;
    gDispatchIdx = DTid;

    uint2 launchIndex = DispatchRaysIndex().xy;
    float2 dims       = float2(DispatchRaysDimensions().xy);
    uint   pixelIdx   = MapPixelID(dims, launchIndex);

    // Load only L1 first
    float3 L1 = load_L1(g_sample_current, pixelIdx);
    float3 accumulation = 0;

    // Also fill "gbuffer" gScratchPing slice 2,3,4...
    /*
    Defined as (0 is general render output):
    - 2: albedo
    - 3: emission, roughness, objID
    - 4: normal
    - 5: position_curr, M_curr
    - 6: position_last
    */
    float3 albedo = (float3)0;
    uint emission = 0u;
    float3 normal = (float3)0;
    float roughness = 0.0f;
    float3 position_curr = (float3)0;
    float3 position_last = (float3)0;
    uint objID = (uint)0;
    uint M_curr = (uint)0;

    if (all(L1 < EPSILON))
    {
        float3 x1 = load_x1(g_sample_current, pixelIdx);
        float3 n1 = load_n1(g_sample_current, pixelIdx);
        float3 o = load_o(g_sample_current, pixelIdx);
        uint matID = load_matID(g_sample_current, pixelIdx);

        Reservoir_DI rdi = loadReservoirDI(g_Reservoirs_last_di, pixelIdx);
        float3 contrib = ReconnectDI(
                             x1, n1, o, matID,
                             rdi.x2_di, rdi.n2_di, rdi.L2_di) * rdi.W_di;

        accumulation = contrib;

        // Set denoiser gbuffer entries
        albedo = materials[matID].Kd.xyz;
        roughness = materials[matID].Pr_Pm_Ps_Pc.x;

        normal = n1;
        position_curr = x1;
        position_last = load_x1(g_sample_last, pixelIdx);

        objID = load_objID(g_sample_current, pixelIdx);
        M_curr = rdi.M_di;
    }
    else
    {
        accumulation = L1;
        emission = 1u;
    }

    // Debug
    /*float3 finalColor = sRGBGammaCorrection(accumulation);
    if (any(isnan(finalColor))) finalColor = float3(1,0,1); // magenta
    if (any(isinf(finalColor))) finalColor = float3(0,1,1); // cyan
    gOutput[uint3(launchIndex, 0)] = float4(finalColor, 1);*/

    //Write to the scratch buffer internally
    gScratchPing[uint3(launchIndex, 0)] = float4(accumulation, 0.0);

    /*Defined as (0 is general render output):
    - 7: albedo
    - 8: emission, roughness, objID
    - 9: normal
    - 10: position_curr, M_curr
    - 1: position_last
    */
    //update the last data
    gScratchPing[uint3(launchIndex, 7)] = gScratchPing[uint3(launchIndex, 2)];
    gScratchPing[uint3(launchIndex, 8)] = gScratchPing[uint3(launchIndex, 3)];
    gScratchPing[uint3(launchIndex, 9)] = gScratchPing[uint3(launchIndex, 4)];
    gScratchPing[uint3(launchIndex, 10)] = gScratchPing[uint3(launchIndex, 5)];
    gScratchPing[uint3(launchIndex, 11)] = gScratchPing[uint3(launchIndex, 6)];
    // write the current data
    gScratchPing[uint3(launchIndex, 2)] = float4(albedo, 0.0f);
    gScratchPing[uint3(launchIndex, 3)] = float4(emission, roughness, objID, 0.0f);
    gScratchPing[uint3(launchIndex, 4)] = float4(normal, 0.0f);
    gScratchPing[uint3(launchIndex, 5)] = float4(position_curr, M_curr);
    gScratchPing[uint3(launchIndex, 6)] = float4(position_last, 0.0f);
}
