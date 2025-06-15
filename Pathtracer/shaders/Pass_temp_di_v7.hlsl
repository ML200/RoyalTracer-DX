#include "Constants_v7.hlsli"
#include "Common_v7.hlsli"
#include "Structures_misc.hlsli"
#include "Motion_vectors_v7.hlsli"
#include "Random_v7.hlsli"
#include "Compression_v7.hlsli"

RWTexture2DArray<float4> gOutput : register(u0);
RWTexture2D<float4> gPermanentData : register(u1);

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
#include "Reservoir_DI_v7.hlsli"
#include "MIS_v7.hlsli"
#include "NEE_Sampling_v7.hlsli"
#include "BSDF_Sampling_v7.hlsli"

[shader("raygeneration")]
void Pass_temp_di_v7() {
    // Get the location within the dispatched 2D grid of work items (often maps to pixels, so this could represent a pixel coordinate).
    uint2 launchIndex = DispatchRaysIndex().xy;
    float2 dims = float2(DispatchRaysDimensions().xy);
    uint pixelIdx = MapPixelID(dims, launchIndex);

    // Load the sample data
    SampleData sdata = loadSampleData(g_sample_current, pixelIdx);
    if(all(sdata.L1 < EPSILON)){
        // Load current reservoir
        Reservoir_DI rdi = loadReservoirDI(g_Reservoirs_current_di, pixelIdx);
        // Get the reprojected pixel position
        uint tempPixelIdx = MapPixelID(dims, GetBestReprojectedPixel_d(sdata.x1, prevView, prevProjection, dims, sdata.objID));
        // Get the reprojected sample data
        SampleData sdata_r = loadSampleData(g_sample_last, tempPixelIdx);
        // Get the reprojected reservoir
        Reservoir_DI rdi_r = loadReservoirDI(g_Reservoirs_last_di, tempPixelIdx);
        // Check wether the reservoir is valid for merge
        bool candidateAcceptedDI =
            (all(sdata_r.L1 < EPSILON) &&
            IsValidReservoir_DI(rdi_r) &&
            !RejectNormal_DI(sdata.n1, sdata_r.n1, 0.5f) &&
            !RejectDistance_DI(sdata.x1, sdata_r.x1, mul(viewI, float4(0, 0, 0, 1)).xyz, 0.1f) &&
            (sdata_r.matID == sdata.matID));

        // Merge the reservoirs
        if(candidateAcceptedDI){
            // Calculate the MIS weights

            // Calculate the reservoirs weights

            // Adjust W of the existing reservoir

            // Update the reservoir

            // Store the merged reservoir
            store_x2_di(rdi.x2_di, g_Reservoirs_current_di, pixelIdx);
            store_n2_di(rdi.n2_di, g_Reservoirs_current_di, pixelIdx);
            store_L2_di(rdi.L2_di, g_Reservoirs_current_di, pixelIdx);
            store_W_di(rdi.W_di, g_Reservoirs_current_di, pixelIdx);
            store_M_di(rdi.M_di, g_Reservoirs_current_di, pixelIdx);
        }
    }
}
