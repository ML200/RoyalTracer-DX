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

RWTexture2DArray<half4> gOutput             : register(u0);
RWTexture2D<half4>      gPermanentData      : register(u1);
RWTexture2DArray<half4> gScratchPing         : register(u8);

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
    uint   pixelIdx   = MapPixelID(DispatchRaysDimensions().xy, launchIndex);

    // Fill "gbuffer" gScratchPing slice 2,3,4...
    /*
    Defined as (0 is general render output):
    - 2: albedo
    - 3: emission, roughness, objID
    - 4: normal
    - 5: position_curr, M_curr
    - 6: position_last
    */
    half3 accumulation;                                      // only live var

    // ---------------------------------------------------------------------
    //  Fast path – pixel has no valid L1 (sky, miss, etc.)
    // ---------------------------------------------------------------------
    float3 L1 = load_L1(g_sample_current, pixelIdx);

    if (all(L1 < EPSILON))
    {
        float3 x1    = load_x1 (g_sample_current, pixelIdx);
        float3 n1    = load_n1 (g_sample_current, pixelIdx);
        float3 o     = load_o  (g_sample_current, pixelIdx);
        uint   matID = load_matID(g_sample_current, pixelIdx);

        Reservoir_DI rdi = loadReservoirDI(g_Reservoirs_last_di, pixelIdx);

        accumulation = ReconnectDI(x1, n1, o, matID,
                                   rdi.x2_di, rdi.n2_di, rdi.L2_di) * rdi.W_di;

        // g-buffer slices – written immediately, no temporaries kept alive
        gScratchPing[uint3(launchIndex, 2)] = half4(materials[matID].Kd.xyz, 0);
        gScratchPing[uint3(launchIndex, 3)] = half4(
                                                0u,
                                                materials[matID].Pr_Pm_Ps_Pc.x,
                                                load_objID(g_sample_current, pixelIdx),
                                                0.0f);
        gScratchPing[uint3(launchIndex, 4)] = half4(n1, 0.0f);
        gScratchPing[uint3(launchIndex, 5)] = half4(x1, rdi.M_di);
        gScratchPing[uint3(launchIndex, 6)] = half4(
                                                load_x1(g_sample_last, pixelIdx), 0.0f);
    }
    // ---------------------------------------------------------------------
    //  Slow path – pixel already has valid radiance in L1
    // ---------------------------------------------------------------------
    else
    {
        accumulation = L1;

        uint matID = load_matID(g_sample_current, pixelIdx);

        gScratchPing[uint3(launchIndex, 2)] = half4(materials[matID].Ke.xyz, 0);
        gScratchPing[uint3(launchIndex, 3)] = half4(
                                                1u,        // emission flag
                                                1.0h,      // roughness (placeholder)
                                                load_objID(g_sample_current, pixelIdx),
                                                0.0f);
        gScratchPing[uint3(launchIndex, 4)] = half4(
                                                load_n1(g_sample_current, pixelIdx), 0.0f);
        gScratchPing[uint3(launchIndex, 5)] = half4(
                                                load_x1(g_sample_current, pixelIdx), 60u);
        gScratchPing[uint3(launchIndex, 6)] = half4(
                                                load_x1(g_sample_last, pixelIdx), 0.0f);
    }

    // ---------------------------------------------------------------------
    //  Common: write final radiance slice
    // ---------------------------------------------------------------------
    gScratchPing[uint3(launchIndex, 0)] = half4(accumulation, 0.0f);

    // DEBUG
    float3 finalColor = sRGBGammaCorrection(accumulation);
    gOutput[uint3(DTid.xy, 0)] = float4(finalColor, 1);
}
