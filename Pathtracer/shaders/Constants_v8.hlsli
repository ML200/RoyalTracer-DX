#define EPSILON 0.00003
#define ONE_MINUS_EPSILON 0x1.fffffep-1f // ~0.99999994
#define SBIAS 0.0008
#define PI 3.1415926535
#define INV_PI 0.3183098861
#define LUT_SIZE 16
#define MIN_NORMAL_INT 0.33f
#define SMOOTH_SPECULAR_THRESHOLD 0.06f
#define kInvalidPixel -1u


//TEXTURES and LUTs
#define SHEEN_LUT_INDEX 0
#define GGX_ESS_LUT_INDEX 1

//DI Reuse
#define TEMP_MCAP_DI 8

#define SPAT_MCAP_DI 48
#define SPAT_EXP_DI 0.85f
#define SPAT_RAD_MAX 48
#define SPAT_RAD_MIN 32

#define SPAT_COUNT_MAX_DI 1

//GI Reuse
#define TEMP_MCAP_GI 8

#define SPAT_MCAP_GI 48
#define SPAT_EXP_GI 0.8f
#define SPAT_RAD_MAX_GI 48
#define SPAT_RAD_MIN_GI 8

#define SPAT_COUNT_MAX_GI 3

// ─── Unified DI+GI reservoir — matID_gi discriminator ────────────────────
// When matID_gi holds one of these sentinels, the sample is a direct-lighting
// candidate that was formerly stored in Reservoir_DI. ReconnectGI branches
// on these values to pick the correct reconnection formula; all x2 BSDF /
// material fetches are skipped for sentinel samples.
//   MATID_ENV_MISS  : x2_gi stores a DIRECTION (not a position).
//                     No geometry term, no x2 BSDF. Jn = 1.
//   MATID_LIGHT_TRI : x2_gi is a world position on an emissive triangle.
//                     n2_s_gi is the light's surface normal. No x2 BSDF;
//                     L2_gi carries the triangle's emission directly.
// Real triangle materials occupy IDs < MATID_LIGHT_TRI.
#define MATID_ENV_MISS  0xFFFFFFFFu
#define MATID_LIGHT_TRI 0xFFFFFFFEu
#define IsSentinelMatID(mid) ((mid) >= MATID_LIGHT_TRI)

#define REUSE_ROUGHNESS_MIN 0.15f
#define REUSE_ROUGHNESS_MAX 0.6f

//Boiling filter (lower strength = less aggressive clamping)
#define GI_BOIL_STRENGTH_TEMP 0.2f
#define GI_BOIL_MIN_AVG_TEMP  1e-8f

#define DI_BOIL_STRENGTH_TEMP 0.2f
#define DI_BOIL_MIN_AVG_TEMP  1e-8f