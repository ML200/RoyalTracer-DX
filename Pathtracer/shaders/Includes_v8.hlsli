//====================================
//INCLUDES V8
//====================================
//compute shaders, #define COMPUTE_PASS before including

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
    //paired reuse textures, per-slot transforms, Lin et al. 2026 §3.2
    //flags bits, 1=flipX 2=flipY 4=transpose
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
    //NRC runtime, see Nrc_v8.hlsli for flag bits
    uint  nrc_flags;
    float nrc_area_spread_c;
    float nrc_lr_scale;
    //[27] explicit pad so float3 below lands on 16-byte boundary
    //HLSL cbuffer rules bump a straddling float3 to next register
    //without this pad, root-constant upload is offset-shifted vs shader reads
    uint   nrc_pad27;
    //scene to [0,1]^3 for tcnn HashGrid, scale_inv = 0.5 / halfExtent
    float3 nrc_scene_center;
    float  nrc_scene_scale_inv;
};

//====================================
//IMAGE SIZE MACROS
//====================================
#define IMG_W (gImageSize.x)
#define IMG_H (gImageSize.y)

#ifdef COMPUTE_PASS
    //emulate DispatchRaysIndex/Dimensions for shared code
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
    float  time;
    float2 jitter;
    float  cameraFar;
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
RWByteAddressBuffer g_pathStateBuffer        : register(u10);

//====================================
//SCENE DATA
//====================================
StructuredBuffer<STriVertex>         BTriVertex          : register(t2);
StructuredBuffer<int>                indices             : register(t1);
RaytracingAccelerationStructure      SceneBVH            : register(t0);
StructuredBuffer<InstanceProperties> instanceProps       : register(t3);
StructuredBuffer<uint>               materialIDs         : register(t4);

//40B compressed AoS material, accessors in Material_Decoder_v8.hlsli
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
//t16/t17/t18 claimed by lookup buffers in LightTree_v8.hlsli

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
//SAMPLING RESERVOIRS RT
//====================================
#include "Path_Sampler_v8.hlsli"
#include "SunSampler_v8.hlsli"
#include "Inline_RT_v8.hlsli"
#include "Reservoir_v8.hlsli"
#include "Path_State_v8.hlsli"

#include "Camera_ray_v8.hlsli"
#include "MIS_v8.hlsli"

//====================================
//DLSS-RR RESOURCES
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
