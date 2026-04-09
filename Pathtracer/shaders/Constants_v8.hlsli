#define EPSILON 0.00003
#define ONE_MINUS_EPSILON 0x1.fffffep-1f // ~0.99999994
#define SBIAS 0.0008
#define PI 3.1415926535
#define INV_PI 0.3183098861
#define LUT_SIZE 16
#define MIN_NORMAL_INT 0.33f
#define SMOOTH_SPECULAR_THRESHOLD 0.06f
#define kInvalidPixel -1u


// TEXTURES & LUTs
#define SHEEN_LUT_INDEX 0
#define GGX_ESS_LUT_INDEX 1

// ___ DI Reuse ___
#define TEMP_MCAP_DI 8

#define SPAT_MCAP_DI 48
#define SPAT_EXP_DI 0.85f
#define SPAT_RAD_MAX 48
#define SPAT_RAD_MIN 32

#define SPAT_COUNT_MAX_DI 8
#define SPAT_COUNT_MIN_DI 8
#define SPAT_TRIS_DI 1


// ___ GI Reuse ___
#define TEMP_MCAP_GI 8

#define SPAT_MCAP_GI 48
#define SPAT_EXP_GI 0.8f
#define SPAT_RAD_MAX_GI 32
#define SPAT_RAD_MIN_GI 32

#define SPAT_COUNT_MAX_GI 8
#define SPAT_COUNT_MIN_GI 8
#define SPAT_TRIS_GI 1

#define REUSE_ROUGHNESS_MIN 0.15f
#define REUSE_ROUGHNESS_MAX 0.6f

// Boiling filter
#define GI_BOIL_STRENGTH_TEMP 0.2f
#define GI_BOIL_MIN_AVG_TEMP  1e-8f

#define DI_BOIL_STRENGTH_TEMP 0.2f
#define DI_BOIL_MIN_AVG_TEMP  1e-8f