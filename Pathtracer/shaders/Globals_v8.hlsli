#pragma once
// Bindings, constants and helpers shared by every pass. The radiance cache buffer g_sharc is
// declared by the front include (Includes_v8.hlsli or IncludesTraining_v8.hlsli) before this file.
#include "SharcLayout.h"
#include "RenderFlags.h"
#define SHARC_DEBUG_MODE ((sharc_enabled >> SHARC_DEBUG_MODE_SHIFT) & SHARC_DEBUG_MODE_MASK)

// Root constants; the order matches Renderer::PopulateCommandList and SHARC_ROOT_CONSTANTS.
cbuffer Push : register(b1)
{
    uint2 gImageSize;
    uint  rs_flags;
    uint  pt_maxBounces;
    uint  pt_rrStartDepth;
    uint  pt_maxDiffuseBounces;
    uint  pt_initialSamples;    // sample count in the low half, the current sample index above
    uint  pt_pointFilter;
    uint  lite_spatMcap;
    uint  lite_spatSlots;
    uint  lite_reuse0;
    uint  lite_reuse1;
    uint  lite_reuse2;
    float lite_normalSimCos;
    float lite_planeDist;
    float lt_cellSize;
    float lt_lodScale;
    uint  lt_now;
    float lt_learnRoughness;
    uint  dbg_dlssLayer;
    uint  dbg_dlssDepthWin;
    float pp_sharpness;
    uint  sharc_enabled;
    uint  sharc_reset;
    uint  sharc_frame;
    float sharc_cellSize;
    float sharc_lodScale;
    uint  sharc_minSamples;
    uint  sharc_historyFrames;
    uint  sharc_maxAge;
    float sharc_queryFootprint;
    uint  sharc_trainBounces;
    uint  sharc_trainRrDepth;
    uint  guide_params;
    uint  lt_bufferBase;        // byte offset of the light-learning region in the cache buffer
    uint  sharc_updateStride;   // training tile width
    float sharc_convergenceThreshold;   // cells less converged than this never end a path (SharcConvergence)
};

#define CLAMP_EMITTERS_MODE ((rs_flags & RS_FLAG_CLAMP_EMITTERS) != 0u)
#define FORCE_DIFFUSE       ((rs_flags & RS_FLAG_FORCE_DIFFUSE) != 0u)

#define RS_FLAG_GUIDE_OFF_DEPTH    0x02000000u
#define RS_FLAG_GUIDE_OFF_MV       0x04000000u
#define RS_FLAG_GUIDE_OFF_NORMALS  0x08000000u
#define RS_FLAG_GUIDE_OFF_ROUGH    0x10000000u
#define RS_FLAG_GUIDE_OFF_ALBEDO   0x20000000u
#define RS_FLAG_GUIDE_OFF_SPECALB  0x40000000u
#define RS_FLAG_GUIDE_OFF_SPECMV   0x80000000u
#define RS_FLAG_GUIDE_OFF_ANY      0xFE000000u

#define IMG_W (gImageSize.x)
#define IMG_H (gImageSize.y)

// The host packs the current sample index into the upper half of the sample count.
#define PT_SAMPLE_COUNT (pt_initialSamples & 0xFFFFu)
#define PT_SAMPLE_INDEX (pt_initialSamples >> 16u)

SamplerState   g_sampler           : register(s0);
SamplerState   g_sampler_LUT       : register(s1);

SamplerState   g_samplerPoint      : register(s3);
Texture2DArray g_LUT         : register(t33);

float4 SampleMaterialTex(Texture2D<float4> tex, float2 uv, float level)
{
    SamplerState s = SamplerDescriptorHeap[(pt_pointFilter != 0u) ? 1u : 0u];
    return tex.SampleLevel(s, uv, level);
}

Texture2D<float4> gSkyStars   : register(t40);

Texture2DArray<float> g_terrainHeightmap : register(t45);

Texture2DArray<float4> g_terrainSurfaceColor : register(t46);
Texture2DArray<float4> g_terrainNormalMap    : register(t47);

Texture2D<float4> g_skyTransmittanceLUT : register(t49);
Texture2D<float4> g_skyMultiScatterLUT  : register(t51);

Texture2D<float> g_blueNoise : register(t59);

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
    float  walltime;

    float3 sceneOriginWorld;

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

    float dofApertureRadius;
    float dofFocusDistance;

    float skyStarIntensity;
    float skyStarGamma;
    float skyStarLodBias;
    float skyStarThreshold;
    float skyNightBaseIntensity;
    // Bruneton atmospheric model.

    float atmos_viewSteps;
    float atmos_lightSteps;
    float atmos_aerialViewSteps;
    float atmos_aerialLightSteps;
    float atmos_multiScatterFactor;
    float atmos_earthShadowSoftness;
    float atmos_haloDistanceKm;
    float planetCenterX;
    float planetCenterY;
    float planetCenterZ;
    float planetRadius;
    float skyGroundY;
    float terrainHeightFrequency;
    float oceanInstanceBase;
    float oceanEnabled;
    // Reconstruction responsivity written per pixel into the denoiser's mask: -1 accumulates the
    // longest, +1 drops history fastest. Ordinary surfaces interpolate between the two ends by
    // roughness - a rough surface's shading barely moves between frames, a smooth one carries a
    // sharp reflection that slides across it. Water gets its own because its sun glitter moves
    // every frame and history that suits a static surface smears it into streaks.
    float dlssResponsivityRough;  // at roughness 1
    float dlssResponsivityMirror; // at roughness 0
    float dlssWaterResponsivity;
}

// Instance-property records from this index up belong to the ocean.
#define OCEAN_ENABLED (oceanEnabled > 0.5f)
#define IS_OCEAN_INSTANCE(instID) (OCEAN_ENABLED && ((instID) >= (uint)oceanInstanceBase))

#define SUN_LATITUDE_DEG    sunLatitude
#define SUN_LONGITUDE_DEG   sunLongitude
#define SUN_DAY_OF_YEAR     sunDayOfYear
#define SUN_SIM_SPEED       sunSimSpeed
#define SUN_START_UTC_HOURS sunStartUTCHours
#define SUN_NIGHT_SPEEDUP   sunNightSpeedup
#define SUN_TURBIDITY       sunTurbidity
#define SUN_INTENSITY_VAL   sunSunIntensity

#define SKY_INTENSITY_VAL   (sunSunIntensity * sunSkyIntensity)
#define SKY_INTENSITY       (sunSunIntensity * sunSkyIntensity)
#define GLOBAL_EMISSION_STRENGTH globalEmissionStrength

#define ATMOS_VIEW_STEPS              ((int)atmos_viewSteps)
#define ATMOS_LIGHT_STEPS             ((int)atmos_lightSteps)
#define ATMOS_AERIAL_VIEW_STEPS       ((int)atmos_aerialViewSteps)
#define ATMOS_AERIAL_LIGHT_STEPS      ((int)atmos_aerialLightSteps)
#define ATMOS_MULTI_SCATTER_FACTOR    atmos_multiScatterFactor
#define ATMOS_EARTH_SHADOW_SOFTNESS   atmos_earthShadowSoftness
#define ATMOS_HALO_DISTANCE_KM        atmos_haloDistanceKm
#define SKY_GROUND_Y                  skyGroundY

inline void SphereToEquiangularFaceUV(float3 dir, out int face, out float2 uv)
{
    float3 a = abs(dir);
    float  ut, vt;
    if (a.x >= a.y && a.x >= a.z) {
        if (dir.x > 0.0f) { face = 0; ut = -dir.z / a.x; vt = -dir.y / a.x; }
        else              { face = 1; ut =  dir.z / a.x; vt = -dir.y / a.x; }
    } else if (a.y >= a.x && a.y >= a.z) {
        if (dir.y > 0.0f) { face = 2; ut =  dir.x / a.y; vt =  dir.z / a.y; }
        else              { face = 3; ut =  dir.x / a.y; vt = -dir.z / a.y; }
    } else {
        if (dir.z > 0.0f) { face = 4; ut =  dir.x / a.z; vt = -dir.y / a.z; }
        else              { face = 5; ut = -dir.x / a.z; vt = -dir.y / a.z; }
    }

    const float kInv = 4.0f / 3.14159265358979f;
    uv = float2(atan(ut), atan(vt)) * kInv * 0.5f + 0.5f;
}

inline float TerrainHeight(float3 dir)
{
    int    face;
    float2 uv;
    SphereToEquiangularFaceUV(dir, face, uv);

    float km = g_terrainHeightmap.SampleLevel(g_sampler_LUT, float3(uv, (float)face), 0.0f);
    return km * 1000.0f;
}

#include "Constants_v8.hlsli"
#include "Common_v8.hlsli"
#include "Data_v8.hlsli"
#include "Random_v8.hlsli"
#include "Compression_v8.hlsli"

#define BLUE_NOISE_SIZE 128u
float BlueNoise(uint2 pixel, uint sampleIndex, uint dim)
{
    const uint  h     = Hash32(dim * 0x9E3779B9u + 0x7F4A7C15u);
    const uint2 texel = (pixel + uint2(h & 0xffu, (h >> 8u) & 0xffu)) & (BLUE_NOISE_SIZE - 1u);
    const float mask  = g_blueNoise.Load(int3((int2)texel, 0));

    const uint  alpha = (dim & 1u) == 0u ? 0xC13FA9A9u : 0x91E10DA6u;
    const float rotation = (float)(sampleIndex * alpha) * 2.3283064365386963e-10f;
    return frac(mask + rotation);
}
float2 BlueNoise2(uint2 pixel, uint sampleIndex, uint dimPair)
{
    return float2(BlueNoise(pixel, sampleIndex, 2u * dimPair),
                  BlueNoise(pixel, sampleIndex, 2u * dimPair + 1u));
}

RWTexture2DArray<float4> gOutput             : register(u0);
RWTexture2D<float4>      gPermanentData      : register(u1);
RWTexture2DArray<float4> gScratchPing        : register(u8);

RWByteAddressBuffer g_liteReservoirs         : register(u4);
RWByteAddressBuffer g_sample_current         : register(u6);
RWByteAddressBuffer g_sample_last            : register(u7);
RWByteAddressBuffer g_pathStateBuffer        : register(u10);
RWByteAddressBuffer g_skyBake                : register(u25);

RWByteAddressBuffer gAutoExpose              : register(u24);
static const uint  AE_OFFS_SUM        = 0u;
static const uint  AE_OFFS_SMOOTHED   = 4u;
static const uint  AE_OFFS_INIT       = 8u;
static const uint  AE_OFFS_TILE_COUNT = 12u;
static const uint  AE_OFFS_PREV_TIME  = 16u;
static const float AE_LOG_OFFSET      = 14.0f;
static const float AE_LOG_SCALE       =  8.0f;

static const uint SENT_OFFS_MASK      = 32u;
static const uint SENT_OFFS_MAXLUMA   = 36u;
static const uint SENT_OFFS_MAXMV     = 40u;
static const uint SENT_OFFS_MAXSPECMV = 44u;
static const uint SENT_OFFS_CAPCOUNT  = 48u;
static const uint SENT_OFFS_BADCOUNT  = 52u;
static const uint SENT_OFFS_FIRSTBAD  = 56u;

StructuredBuffer<STriVertex>         BTriVertex          : register(t2);
StructuredBuffer<int>                indices             : register(t1);
RaytracingAccelerationStructure      SceneBVH            : register(t0);
StructuredBuffer<InstanceProperties> instanceProps       : register(t3);
StructuredBuffer<uint>               materialIDs         : register(t4);

StructuredBuffer<MatPacked>          g_mat               : register(t5);

StructuredBuffer<LightTriangle>      g_EmissiveTriangles : register(t6);
StructuredBuffer<uint>               gTriToLightId       : register(t15);

StructuredBuffer<uint4> gLT_TLAS                  : register(t9);
StructuredBuffer<uint4> gLT_BLAS                  : register(t10);
StructuredBuffer<BlasRangeGpu>     gLT_Range        : register(t11);
Buffer<uint>                       gLT_LeafTriIndex : register(t12);

StructuredBuffer<LightSlotGpu>     gLT_Slot         : register(t7);

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

#include "SunSampler_v8.hlsli"
#include "Ocean_v8.hlsli"
#include "OceanMediumState.hlsli"
#include "Inline_RT_v8.hlsli"
#include "Material_SSS_v8.hlsli"
#include "Path_State_v8.hlsli"

#include "Camera_ray_v8.hlsli"
#include "PsrGuide_v8.hlsli"

// Denoiser guides, written by the compute passes.
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
RWTexture2D<float>  g_dlssResponsivity   : register(u26);
