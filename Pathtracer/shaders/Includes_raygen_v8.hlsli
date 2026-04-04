cbuffer Push : register(b1)
{
    uint2 gImageSize;          // [0-1]
    uint  g_InputStackIdx;     // [2]
    uint  g_OutputStackIdx;    // [3]
    uint  rs_tempMcapDI;       // [4]
    uint  rs_tempMcapGI;       // [5]
    uint  rs_spatCountMaxDI;   // [6]
    uint  rs_spatCountMinDI;   // [7]
    uint  rs_spatRadMaxDI;     // [8]
    uint  rs_spatRadMinDI;     // [9]
    uint  rs_spatCountMaxGI;   // [10]
    uint  rs_spatCountMinGI;   // [11]
    uint  rs_spatRadMaxGI;     // [12]
    uint  rs_spatRadMinGI;     // [13]
    uint  rs_flags;            // [14] bit0=tempDI, bit1=tempGI, bit2=spatDI, bit3=spatGI
    float rs_reuseRoughnessMin;// [15]
    float rs_reuseRoughnessMax;// [16]
    uint  _pad17;              // [17]
    uint  _pad18;              // [18]
    uint  _pad19;              // [19]
};

#define ENABLE_RAY_QUERY_INLINE // Activate support for inline ray tracing

SamplerState g_sampler : register(s0);

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

cbuffer CameraParams : register(b0)
{
    float4x4 view;
    float4x4 projection;
    float4x4 viewI;
    float4x4 projectionI;
    float4x4 prevView;
    float4x4 prevProjection;
    float time;
    float2 jitter;
    float _cbpad0;
    // SunSettings (runtime-editable)
    float sunLatitude;
    float sunLongitude;
    float sunDayOfYear;
    float sunSimSpeed;
    float sunStartUTCHours;
    float sunNightSpeedup;
    float sunTurbidity;
    float sunSunIntensity;
    float sunSkyIntensity;
    float _sunpad0, _sunpad1, _sunpad2;
}

// Override sun #defines with cbuffer values (before SunSampler includes them)
#define SUN_LATITUDE_DEG    sunLatitude
#define SUN_LONGITUDE_DEG   sunLongitude
#define SUN_DAY_OF_YEAR     sunDayOfYear
#define SUN_SIM_SPEED       sunSimSpeed
#define SUN_START_UTC_HOURS sunStartUTCHours
#define SUN_NIGHT_SPEEDUP   sunNightSpeedup
#define SUN_TURBIDITY       sunTurbidity
#define SUN_INTENSITY_VAL   sunSunIntensity
#define SKY_INTENSITY_VAL   sunSkyIntensity
#define SKY_INTENSITY       sunSkyIntensity

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


// Convenience macros (match Includes_v8 for shared code)
#define IMG_W (gImageSize.x)
#define IMG_H (gImageSize.y)

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

// These includes need access to ALL previous buffers
#include "Path_Sampler_v8.hlsli"
#include "SunSampler_v8.hlsli"
#include "Clouds_v8.hlsli"
#include "Reservoir_DI_v8.hlsli"
#include "Reservoir_GI_v8.hlsli"
#include "PayloadPath_v8.hlsli"
#include "Inline_RT_v8.hlsli"
#include "Camera_ray_v8.hlsli"
#include "VolumeStackPacked_v8.hlsli"
#include "MIS_v8.hlsli"