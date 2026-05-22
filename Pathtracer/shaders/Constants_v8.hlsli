//====================================
//NUMERIC CONSTANTS
//====================================
#define EPSILON 0.00003
#define ONE_MINUS_EPSILON 0x1.fffffep-1f
#define SBIAS 0.0008
#define PI 3.1415926535
#define INV_PI 0.3183098861
#define LUT_SIZE 16
#define MIN_NORMAL_INT 0.33f
//BSDF delta lobe gate, low so VNDF sampling stays unbiased
#define SMOOTH_SPECULAR_THRESHOLD 0.06f
//NRC cache gate, well above SH deg-4 representable lobe width
#define NRC_CACHE_ROUGHNESS_MIN 0.25f
//Training-emission gate, deliberately stricter than the inference gate so the
//cache is taught only on safely diffuse-dominated samples. Low-roughness
//dielectrics produce L_s/(alpha+beta) targets amplified ~25x by the small
//Fresnel reflSum, which clamp to kTargetMax and burn in as phantom
//reflections that never fade. Inference still queries down to 0.25 so the
//cache covers slightly glossier surfaces with the cleanly-trained predictor.
#define NRC_TRAIN_ROUGHNESS_MIN 0.4f
#define kInvalidPixel -1u

//====================================
//RAY TMAX
//====================================
//Planet wide ray range. Earth diameter is ~1.27e7 m; this leaves headroom
//for tangent grazing rays from orbital altitudes and for second bounces
//that escape upward into space before being marked as misses. Used for
//primary, reflection, and bounce rays in Pass_raygen_v8.hlsl. The sky
//depth fallback in Pass_shading_v8.hlsl reads cameraFar (CPU side),
//which should stay >= this value or distant surface hits appear closer
//than the sky in the DLSS RR depth buffer.
#define RAY_TMAX_PLANET 1e9f

//====================================
//ATMOSPHERE RING DEBUG (temporary)
//====================================
//Localises the nadir-centred ring artifact. Set non-zero, rebuild, observe,
//report. Ship as 0. Remove the #if blocks in Pass_clouds_primary_v8,
//Pass_shading_v8 and Inline_RT_v8 once the cause is found.
//  1 = cloud-pass false colour: R=combinedTr, G=unifiedInscatter
//  2 = shading false colour:    R=indirect/GI, G=reflection, B=direct
//  3 = sky fully OFF: EvaluateSky / EvaluateSkyBackground / *Behind and the
//      atmosphere march all return black. Tests whether the ring is the sky.
//  4 = planet shading normal (sv.n_s) shown as RGB. Tests for bad normals.
#define ATM_DEBUG_RING 0


//====================================
//TEXTURE AND LUT INDICES
//====================================
#define SHEEN_LUT_INDEX 0
#define GGX_ESS_LUT_INDEX 1

//====================================
//REUSE CAPS
//====================================
#define TEMP_MCAP 8

#define SPAT_MCAP 100000
#define SPAT_EXP  0.8f
#define SPAT_RAD_MAX 48
#define SPAT_RAD_MIN 8

#define SPAT_COUNT_MAX 3

//====================================
//RESERVOIR MATID SENTINELS
//====================================
//env miss x2 is direction, light tri x2 is triangle position, real mats sit below
#define MATID_ENV_MISS    0xFFFFFFFFu
#define MATID_LIGHT_TRI   0xFFFFFFFEu
#define IsSentinelMatID(mid) ((mid) >= MATID_LIGHT_TRI)

//====================================
//PLANET TERRAIN (Phase 5)
//====================================
//Instance-ID base for terrain in the unified TLAS - streamed chunks + the
//6-face fallback layer. MUST match planet::TERRAIN_INSTANCE_BASE
//(rdn/planet/stream_orchestrator.h). Scene mesh instances keep InstanceID
//0..N-1; any instID >= this is terrain and is shaded procedurally.
#define TERRAIN_INSTANCE_BASE 0x100000u
//Sentinel matID for terrain. Sits just below the env/light-tri sentinels so
//IsSentinelMatID() stays false - terrain is a normal shadable opaque surface,
//it just gets a procedural material instead of a g_mat entry.
#define MATID_TERRAIN 0xFFFFFFFDu
//First-cut fixed terrain material (procedural slope/altitude shading is later).
#define TERRAIN_ALBEDO    float3(0.42f, 0.36f, 0.30f)
#define TERRAIN_ROUGHNESS 0.92f
//Chunk tessellation grid (quads per edge). MUST match planet::CHUNK_GRID
//(rdn/planet/chunk_mesh.h) - the terrain shader decodes primID against it.
#define TERRAIN_CHUNK_GRID 128u

//====================================
//ROUGHNESS REUSE GATE
//====================================
#define REUSE_ROUGHNESS_MIN 0.15f
#define REUSE_ROUGHNESS_MAX 0.6f

//====================================
//BOILING FILTER
//====================================
//lower strength clamps less
#define BOIL_STRENGTH_TEMP 0.2f
#define BOIL_MIN_AVG_TEMP  1e-8f
