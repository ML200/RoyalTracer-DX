#include "Motion_vectors_v7.hlsli"

RWTexture2DArray<float4> gOutput : register(u0);
RWTexture2D<float4> gPermanentData : register(u1);

RWByteAddressBuffer g_sample_current : register(u6);
/*RWStructuredBuffer<SampleData> g_sample_last : register(u7);
RWStructuredBuffer<Reservoir_DI> g_Reservoirs_current : register(u2);
RWStructuredBuffer<Reservoir_DI> g_Reservoirs_last : register(u3);
RWStructuredBuffer<Reservoir_GI> g_Reservoirs_current_gi : register(u4);
RWStructuredBuffer<Reservoir_GI> g_Reservoirs_last_gi : register(u5);

StructuredBuffer<STriVertex> BTriVertex : register(t2);
StructuredBuffer<int> indices : register(t1);
RaytracingAccelerationStructure SceneBVH : register(t0);
StructuredBuffer<InstanceProperties> instanceProps : register(t3);
StructuredBuffer<uint> materialIDs : register(t4);
StructuredBuffer<Material> materials : register(t5);
StructuredBuffer<LightTriangle> g_EmissiveTriangles : register(t6);*/

#include "Sample_data.hlsli"

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

[shader("raygeneration")]
void Pass_shading_v7() {
    uint2 launchIndex = DispatchRaysIndex().xy;
    float2 dims       = float2(DispatchRaysDimensions().xy);
    uint pixelIdx     = MapPixelID(dims, launchIndex);

    gOutput[uint3(launchIndex, 0)] = float4(1.0f,0.0f,1.0f, 1.0f);
}
