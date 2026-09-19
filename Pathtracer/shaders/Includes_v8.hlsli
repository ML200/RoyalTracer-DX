#ifndef INCLUDES_V8_HLSLI
#define INCLUDES_V8_HLSLI
#include "SharcLayout.h"
#include "RenderFlags.h"
#define SHARC_DEBUG_MODE ((sharc_enabled >> SHARC_DEBUG_MODE_SHIFT) & SHARC_DEBUG_MODE_MASK)

#define DISABLE_ALPHA_TEST 0

// Push constant order must match the host layout.
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

    float temp_normalSimCos;
    float temp_planeDist;
    float temp_jacClamp;

    float ucw_clampMax;

    float rs_reconnectRoughnessMin;

    float rs_reconnectDistMin;

    uint  rs_rcMaxK;

    float rs_rcFpKappa;

    float rs_corrReductionPow;

    float rs_rejNormalDot;
    float rs_rejDistance;

    float spmis_normalFuzz;
    uint  spmis_normalBits;
    float spmis_searchR0;
    float spmis_searchGrow;
    uint  spmis_searchIters;

    uint  pt_maxDiffuseBounces;

    uint  dbg_dlssLayer;
    uint  dbg_dlssDepthWin;

    uint  spmis_reuseN;
    uint  spmis_risN;
    uint  spmis_mcap;
    uint  spmis_tileSize;
    float spmis_jacThreshold;
    float spmis_normalSimCos;

    uint  pt_maxBounces;
    uint  pt_rrStartDepth;

    uint  pt_initialSamples;

    float spmis_planeDist;

    uint  pt_pointFilter;

    float pp_sharpness;

    uint  sharc_enabled;
    uint  sharc_reset;
    uint  sharc_frame;
    uint  sharc_updateStride;
    float sharc_cellSize;
    float sharc_lodScale;
    uint  sharc_minSamples;
    uint  sharc_historyFrames;
    uint  sharc_maxAge;
    float sharc_queryFootprint;
    uint  sharc_trainBounces;
    uint  sharc_trainRrDepth;

    uint  guide_params;
};

#define RS_FLAG_CLAMP_EMITTERS  0x100u
#define CLAMP_EMITTERS_MODE  ((rs_flags & RS_FLAG_CLAMP_EMITTERS) != 0u)

#define RS_FLAG_SPMIS_SPATIAL  0x10u
#define SPMIS_SPATIAL_MODE  ((rs_flags & RS_FLAG_SPMIS_SPATIAL) != 0u)

#define RS_FLAG_SPMIS_CONF_ADJUST  0x2000u
#define SPMIS_CONF_ADJUST  ((rs_flags & RS_FLAG_SPMIS_CONF_ADJUST) != 0u)

#define RS_FLAG_DISABLE_CORR_REDUCTION  0x40u
#define CORR_REDUCTION_OFF  ((rs_flags & RS_FLAG_DISABLE_CORR_REDUCTION) != 0u)

#define RS_FLAG_FORCE_DIFFUSE  0x20000u
#define FORCE_DIFFUSE  ((rs_flags & RS_FLAG_FORCE_DIFFUSE) != 0u)

#define RS_FLAG_SPMIS_CELL_JITTER  0x40000u
#define SPMIS_CELL_JITTER  ((rs_flags & RS_FLAG_SPMIS_CELL_JITTER) != 0u)

#define RS_FLAG_NO_SPEC_REPROJ  0x200u
#define NO_SPEC_REPROJ  ((rs_flags & RS_FLAG_NO_SPEC_REPROJ) != 0u)

#define RS_FLAG_NO_REUSE_VIS  0x800u
#define REUSE_VIS_OFF  ((rs_flags & RS_FLAG_NO_REUSE_VIS) != 0u)

#define RS_FLAG_NO_FINAL_VIS  0x1000u
#define FINAL_VIS_OFF  ((rs_flags & RS_FLAG_NO_FINAL_VIS) != 0u)

#define RS_FLAG_HYBRID_SHIFT  0x4000u
#define HYBRID_SHIFT_ON  ((rs_flags & RS_FLAG_HYBRID_SHIFT) != 0u)

#define RS_FLAG_RC_FOOTPRINT  0x100000u
#define RC_FOOTPRINT_ON  ((rs_flags & RS_FLAG_RC_FOOTPRINT) != 0u)

#define RS_FLAG_DUAL_MV  0x200000u
#define DUAL_MV_ON  ((rs_flags & RS_FLAG_DUAL_MV) != 0u)

#define RS_FLAG_PT_ONLY  0x1000000u
#define PT_ONLY_MODE  ((rs_flags & RS_FLAG_PT_ONLY) != 0u)

#define RS_FLAG_RGB_SHADE  0x400000u
#define RGB_SHADE_ON  ((rs_flags & RS_FLAG_RGB_SHADE) != 0u)

#define RS_FLAG_LOBE_PSS  0x800000u
#define LOBE_PSS_ON  (((rs_flags & RS_FLAG_LOBE_PSS) != 0u) && HYBRID_SHIFT_ON)

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

#ifdef COMPUTE_PASS

    #define gImageWidth  (gImageSize.x)
    #define gImageHeight (gImageSize.y)
    #define DispatchRaysDimensions() uint3(gImageWidth, gImageHeight, 1)
    static uint3 gDispatchIdx;
    #define DispatchRaysIndex()      gDispatchIdx
#endif

#define ENABLE_RAY_QUERY_INLINE

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
    float planetCenterX;
    float planetCenterY;
    float planetCenterZ;
    float planetRadius;
    float skyGroundY;
    float terrainHeightFrequency;
    float cloudEnabled;
    float cloudCoverage;
    float cloudBaseKm;
    float cloudThicknessKm;
    float cloudScale;
    float cloudExtinction;
    float cloudDetail;
    float cloudWindX;
    float cloudWindZ;
    float cloudMultipleScattering;
    float cloudAmbient;
    float cloudViewSteps;
    float cloudReflectionSteps;
    float cloudSeed;
    float cloudDebugView;
    float cloudGuideThreshold;
    float cloudLightingSamples;
    float cloudFineDetail;
    float cloudDeltaSeconds;
    float cloudDensityCache;
    uint cloudDensityEpoch;
}

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

RWByteAddressBuffer g_sample_current         : register(u6);
RWByteAddressBuffer g_sample_last            : register(u7);
RWByteAddressBuffer g_Reservoirs_current     : register(u4);
RWByteAddressBuffer g_Reservoirs_last        : register(u5);

RWByteAddressBuffer g_raygenQueue            : register(u26);

#ifdef SPMIS_GRID_NONCOHERENT
RWByteAddressBuffer g_pathStateBuffer        : register(u10);
RWByteAddressBuffer g_spmisBuffer            : register(u25);
#else

globallycoherent RWByteAddressBuffer g_pathStateBuffer        : register(u10);

globallycoherent RWByteAddressBuffer g_spmisBuffer            : register(u25);
#endif

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

#include "Path_Sampler_v8.hlsli"
#include "SunSampler_v8.hlsli"
#include "Inline_RT_v8.hlsli"
#include "Material_SSS_v8.hlsli"
#include "Reservoir_v8.hlsli"
#include "Path_State_v8.hlsli"
#include "HashGridHash_v8.hlsli"

#include "Camera_ray_v8.hlsli"
#include "MIS_v8.hlsli"
#include "PsrGuide_v8.hlsli"

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

#ifdef COMPUTE_PASS
inline float ResolveReuseVis(uint pixelIdx, Reservoir r, float3 contrib)
{
    if (!REUSE_VIS_OFF || FINAL_VIS_OFF || GetPHat(contrib) <= 0.0f) return 1.0f;
    const float3        x1 = load_x1(g_sample_current, pixelIdx);
    const SurfaceVertex sv = BuildVertex(g_sample_current, pixelIdx, x1, InitOrigin());

    return Luma(ReconnectVis(sv.x, sv.n_s, r.matID, r.x2, r.n2_s));
}

#endif

#endif
