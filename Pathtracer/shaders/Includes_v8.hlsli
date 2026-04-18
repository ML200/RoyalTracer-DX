//Includes_v8.hlsli
//Compute shaders:  #define COMPUTE_PASS before including.

#ifndef INCLUDES_V8_HLSLI
#define INCLUDES_V8_HLSLI

//Push constants
cbuffer Push : register(b1)
{
    uint2 gImageSize;          // [0-1]
    uint  g_InputStackIdx;     // [2]
    uint  g_OutputStackIdx;    // [3]
    uint  rs_tempMcap;         // [4]
    uint  rs_spatCountMax;     // [5]
    uint  rs_spatCountMin;     // [6]
    uint  rs_spatRadMax;       // [7]
    uint  rs_spatRadMin;       // [8]
    uint  rs_flags;            // [9]  bit0=temp, bit1=spat
    float rs_reuseRoughnessMin;// [10]
    float rs_reuseRoughnessMax;// [11]
    uint  rs_spatTries;        // [12]
    // Paired reuse textures, per-slot randomized transforms (Lin et al. 2026 §3.2)
    // flags bits: 1=flipX, 2=flipY, 4=transpose
    uint  rs_reuseOffset0_x;   // [13]
    uint  rs_reuseOffset0_y;   // [14]
    uint  rs_reuseFlags0;      // [15]
    uint  rs_reuseOffset1_x;   // [16]
    uint  rs_reuseOffset1_y;   // [17]
    uint  rs_reuseFlags1;      // [18]
    uint  rs_reuseOffset2_x;   // [19]
    uint  rs_reuseOffset2_y;   // [20]
    uint  rs_reuseFlags2;      // [21]
    // Neighbor-rejection thresholds (spatial select pass)
    float rs_rejNormalDot;     // [22]  dot(n_A, n_B) must be >= this
    float rs_rejDistance;      // [23]  |proj-on-normal| distance gate (world units)
};

//Image size convenience macros
#define IMG_W (gImageSize.x)
#define IMG_H (gImageSize.y)

#ifdef COMPUTE_PASS
    //Compute shaders emulate DispatchRaysIndex/Dimensions for shared code
    #define gImageWidth  (gImageSize.x)
    #define gImageHeight (gImageSize.y)
    #define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)
    static uint3 gDispatchIdx;
    #define DispatchRaysIndex()      gDispatchIdx
#endif

//Inline ray tracing support
#define ENABLE_RAY_QUERY_INLINE

//Samplers & LUTs
SamplerState   g_sampler     : register(s0);
SamplerState   g_sampler_LUT : register(s1);
Texture2DArray g_LUT         : register(t33);

//Camera
cbuffer CameraParams : register(b0)
{
    float4x4 view;
    float4x4 projection;
    float4x4 viewI;
    float4x4 projectionI;
    float4x4 prevView;
    float4x4 prevProjection;
    float  time;
    float2 jitter;
    float  _cbpad0;
    // Sun settings
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

//Core utility headers
#include "Constants_v8.hlsli"
#include "Common_v8.hlsli"
#include "Data_v8.hlsli"
#include "Random_v8.hlsli"
#include "Compression_v8.hlsli"

//Output / scratch / reservoir buffers
RWTexture2DArray<float4> gOutput             : register(u0);
RWTexture2D<float4>      gPermanentData      : register(u1);
RWTexture2DArray<float4> gScratchPing        : register(u8);

RWByteAddressBuffer g_sample_current         : register(u6);
RWByteAddressBuffer g_sample_last            : register(u7);
RWByteAddressBuffer g_Reservoirs_current     : register(u4);
RWByteAddressBuffer g_Reservoirs_last        : register(u5);
RWByteAddressBuffer g_pathStateBuffer        : register(u10);

//Scene data
StructuredBuffer<STriVertex>         BTriVertex          : register(t2);
StructuredBuffer<int>                indices             : register(t1);
RaytracingAccelerationStructure      SceneBVH            : register(t0);
StructuredBuffer<InstanceProperties> instanceProps       : register(t3);
StructuredBuffer<uint>               materialIDs         : register(t4);

// Material buffer — compressed AoS, 40 B / material (was 112 B).
// See Data_v8.hlsli for layout and Material_Decoder_v8.hlsli for accessors.
StructuredBuffer<MatPacked>          g_mat               : register(t5);

StructuredBuffer<LightTriangle>      g_EmissiveTriangles : register(t6);
StructuredBuffer<uint>               gTriToLightId       : register(t15);

//Light tree
StructuredBuffer<LightTLASNodeGpu> gLT_TLAS         : register(t9);
StructuredBuffer<LightBLASNodeGpu> gLT_BLAS         : register(t10);
StructuredBuffer<BlasRangeGpu>     gLT_Range        : register(t11);
Buffer<uint>                       gLT_LeafTriIndex : register(t12);
Buffer<float>                      gLT_LeafAliasProb: register(t16);
Buffer<uint>                       gLT_LeafAliasIdx : register(t17);

//Shading and material headers
#include "Material_Decoder_v8.hlsli"
#include "LightTree_v8.hlsli"
#include "Sample_Data_v8.hlsli"
#include "Fresnel_v8.hlsli"
#include "Material_Common_v8.hlsli"
#include "Material_GGX_v8.hlsli"
#include "Material_Lambertian_v8.hlsli"
#include "Material_Coat_v8.hlsli"
#include "Material_Sheen_v8.hlsli"
#include "BXDF_v8.hlsli"

//Sampling, reservoirs, ray tracing
#include "Path_Sampler_v8.hlsli"
#include "SunSampler_v8.hlsli"
#include "Clouds_v8.hlsli"
#include "Inline_RT_v8.hlsli"
#include "Reservoir_v8.hlsli"
#include "Path_State_v8.hlsli"

#include "Camera_ray_v8.hlsli"
#include "MIS_v8.hlsli"

//DLSS-RR resources
#ifdef COMPUTE_PASS
RWTexture2D<float>  g_dlssDepth          : register(u11);
RWTexture2D<float2> g_dlssMVec           : register(u12);
RWTexture2D<float4> g_dlssNormals        : register(u13);
RWTexture2D<float4> g_dlssDiffuseAlbedo  : register(u14);
RWTexture2D<float4> g_dlssOutput         : register(u15);
RWTexture2D<float4> g_dlssSpecularAlbedo : register(u16);
RWTexture2D<float>  g_dlssRoughness      : register(u17);
RWTexture2D<float2> g_dlssSpecMVec       : register(u18);
RWTexture2D<float>  g_dlssSpecHitDist    : register(u19);
RWTexture2D<float4> g_dlssTransparency   : register(u20);
RWTexture2D<float4> g_dlssColorPreTrans  : register(u21);
RWTexture2D<float4> g_dlssInput          : register(u22);
RWTexture2D<float>  g_dlssBiasHint       : register(u23);
#endif

#endif // INCLUDES_V8_HLSLI
