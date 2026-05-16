//====================================
//INCLUDES V8
//====================================
//compute shaders must define COMPUTE_PASS before include

#ifndef INCLUDES_V8_HLSLI
#define INCLUDES_V8_HLSLI

//====================================
//PUSH CONSTANTS
//====================================
cbuffer Push : register(b1)
{
    uint2 gImageSize;
    uint  g_InputStackIdx;
    uint  g_OutputStackIdx;
    uint  rs_tempMcap;
    uint  rs_spatCountMax;
    uint  rs_spatCountMin;
    uint  rs_spatRadMax;
    uint  rs_spatRadMin;
    uint  rs_flags;
    float rs_reuseRoughnessMin;
    float rs_reuseRoughnessMax;
    uint  rs_spatTries;
    //paired reuse textures, flag bits 1=flipX 2=flipY 4=transpose
    uint  rs_reuseOffset0_x;
    uint  rs_reuseOffset0_y;
    uint  rs_reuseFlags0;
    uint  rs_reuseOffset1_x;
    uint  rs_reuseOffset1_y;
    uint  rs_reuseFlags1;
    uint  rs_reuseOffset2_x;
    uint  rs_reuseOffset2_y;
    uint  rs_reuseFlags2;
    //neighbor rejection thresholds for spatial select
    float rs_rejNormalDot;
    float rs_rejDistance;
    //NRC runtime, flag bits in Nrc_v8.hlsli
    uint  nrc_flags;
    float nrc_area_spread_c;
    float nrc_lr_scale;
    //dynamic per-frame cap on inference slots, sized from prior frame's actual
    //counter via async readback. Both raygen (NrcAppendInference cap) and the
    //CUDA inference dispatch (count) read this so the work matches demand
    //instead of paying for the buffer's static 2*W*H capacity every frame.
    //Slot 27 also doubles as the 16B alignment pad for nrc_scene_center below.
    uint  nrc_inference_capacity;
    //scene to [0,1]^3 for tcnn HashGrid, scale_inv = 0.5/halfExtent
    float3 nrc_scene_center;
    float  nrc_scene_scale_inv;
};

//====================================
//IMAGE SIZE MACROS
//====================================
#define IMG_W (gImageSize.x)
#define IMG_H (gImageSize.y)

#ifdef COMPUTE_PASS
    //emulate DispatchRaysIndex/Dimensions
    #define gImageWidth  (gImageSize.x)
    #define gImageHeight (gImageSize.y)
    #define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)
    static uint3 gDispatchIdx;
    #define DispatchRaysIndex()      gDispatchIdx
#endif

#define ENABLE_RAY_QUERY_INLINE

//====================================
//SAMPLERS AND LUTS
//====================================
SamplerState   g_sampler     : register(s0);
SamplerState   g_sampler_LUT : register(s1);
Texture2DArray g_LUT         : register(t33);

//====================================
//CAMERA
//====================================
cbuffer CameraParams : register(b0)
{
    float4x4 view;
    float4x4 projection;
    float4x4 viewI;
    float4x4 projectionI;
    float4x4 prevView;
    float4x4 prevProjection;
    float  time;       //jitter frame index cast to float, drives random seeds and the sun
    float2 jitter;
    float  cameraFar;
    float  walltime;   //accumulated wall-clock seconds, drives dt-based auto-exposure
    float3 _camPad0;   //matches the host's 3-float pad inside extra[8]
    //sun settings
    float sunLatitude;
    float sunLongitude;
    float sunDayOfYear;
    float sunSimSpeed;
    float sunStartUTCHours;
    float sunNightSpeedup;
    float sunTurbidity;
    float sunSunIntensity;
    float sunSkyIntensity;
    float globalEmissionStrength;
    //thin-lens DoF, driven from the Camera class on the host side
    float dofApertureRadius;
    float dofFocusDistance;
}

#define SUN_LATITUDE_DEG    sunLatitude
#define SUN_LONGITUDE_DEG   sunLongitude
#define SUN_DAY_OF_YEAR     sunDayOfYear
#define SUN_SIM_SPEED       sunSimSpeed
#define SUN_START_UTC_HOURS sunStartUTCHours
#define SUN_NIGHT_SPEEDUP   sunNightSpeedup
#define SUN_TURBIDITY       sunTurbidity
#define SUN_INTENSITY_VAL   sunSunIntensity
//Sky integrates atmospheric in-scatter in units of ATMOS_SOLAR_IRRADIANCE
//(=1 in the model). For the sky to read in the same scene units as the path-
//traced sun, the final scale must equal the sun's irradiance. sunSkyIntensity
//therefore acts as a multiplicative artistic boost on top of physical (1.0 =
//physically calibrated, >1 = sky pops more than ground, <1 = muted sky).
#define SKY_INTENSITY_VAL   (sunSunIntensity * sunSkyIntensity)
#define SKY_INTENSITY       (sunSunIntensity * sunSkyIntensity)
#define GLOBAL_EMISSION_STRENGTH globalEmissionStrength

//====================================
//CORE UTILITY HEADERS
//====================================
#include "Constants_v8.hlsli"
#include "Common_v8.hlsli"
#include "Data_v8.hlsli"
#include "Random_v8.hlsli"
#include "Compression_v8.hlsli"

//====================================
//OUTPUT SCRATCH RESERVOIR BUFFERS
//====================================
RWTexture2DArray<float4> gOutput             : register(u0);
RWTexture2D<float4>      gPermanentData      : register(u1);
RWTexture2DArray<float4> gScratchPing        : register(u8);

RWByteAddressBuffer g_sample_current         : register(u6);
RWByteAddressBuffer g_sample_last            : register(u7);
RWByteAddressBuffer g_Reservoirs_current     : register(u4);
RWByteAddressBuffer g_Reservoirs_last        : register(u5);
//globallycoherent so the raygen scratch+reload pattern (tpost, nrcA0)
//actually drops those values from live state across the bounce-loop
//TraceRay reorder boundary. Without it, DXC store-to-load forwards the
//scratch values and keeps them register-resident across the trace,
//defeating the spill entirely. Spat GI pass takes a small cache hit on
//its cross-pixel reads; revert if that shows up in profiling.
globallycoherent RWByteAddressBuffer g_pathStateBuffer        : register(u10);

//====================================
//AUTO EXPOSURE STATE (20 B persistent)
//====================================
//[0]  uint  AE_OFFS_SUM        — fixed-point log2-luminance sum (cleared each frame)
//[4]  float AE_OFFS_SMOOTHED   — temporally smoothed log2-luminance (read by postprocess)
//[8]  uint  AE_OFFS_INIT       — first-frame flag, 0 = uninitialized, 1 = ready
//[12] uint  AE_OFFS_TILE_COUNT — contributing tile count (cleared each frame)
//[16] float AE_OFFS_PREV_TIME  — last frame's CameraParams.time, drives dt-based smoothing
//fixed-point packing: store uint((log2Lum + AE_LOG_OFFSET) * AE_LOG_SCALE) per group.
//log2 units are clamped to [-AE_LOG_OFFSET, +AE_LOG_OFFSET] before packing so the sum
//stays inside uint32 at 4K: tilesAt4K(32400) * 2*OFFSET*SCALE = 32400*224 ≈ 7.3M, safe.
RWByteAddressBuffer gAutoExpose              : register(u24);
static const uint  AE_OFFS_SUM        = 0u;   // fixed-point log2-lum sum (cleared each frame)
static const uint  AE_OFFS_SMOOTHED   = 4u;   // float, persistent
static const uint  AE_OFFS_INIT       = 8u;   // uint flag, persistent
static const uint  AE_OFFS_TILE_COUNT = 12u;  // uint, contributing tile count (cleared each frame)
static const uint  AE_OFFS_PREV_TIME  = 16u;  // float, persistent
static const float AE_LOG_OFFSET      = 14.0f;
static const float AE_LOG_SCALE       =  8.0f;

//====================================
//SCENE DATA
//====================================
StructuredBuffer<STriVertex>         BTriVertex          : register(t2);
StructuredBuffer<int>                indices             : register(t1);
RaytracingAccelerationStructure      SceneBVH            : register(t0);
StructuredBuffer<InstanceProperties> instanceProps       : register(t3);
StructuredBuffer<uint>               materialIDs         : register(t4);

//40B compressed AoS, accessors in Material_Decoder_v8.hlsli
StructuredBuffer<MatPacked>          g_mat               : register(t5);

StructuredBuffer<LightTriangle>      g_EmissiveTriangles : register(t6);
StructuredBuffer<uint>               gTriToLightId       : register(t15);

//====================================
//LIGHT TREE
//====================================
StructuredBuffer<LightTLASNodeGpu> gLT_TLAS         : register(t9);
StructuredBuffer<LightBLASNodeGpu> gLT_BLAS         : register(t10);
StructuredBuffer<BlasRangeGpu>     gLT_Range        : register(t11);
Buffer<uint>                       gLT_LeafTriIndex : register(t12);
//t16/t17/t18 used by LightTree_v8.hlsli lookup buffers

//====================================
//SHADING AND MATERIAL HEADERS
//====================================
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

//====================================
//SAMPLING AND RESERVOIRS
//====================================
#include "Path_Sampler_v8.hlsli"
#include "SunSampler_v8.hlsli"
#include "Inline_RT_v8.hlsli"
#include "Reservoir_v8.hlsli"
#include "Path_State_v8.hlsli"

#include "Camera_ray_v8.hlsli"
#include "MIS_v8.hlsli"

//====================================
//DLSS RR RESOURCES
//====================================
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

#endif
