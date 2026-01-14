#define EPSILON 0.000003
#define ONE_MINUS_EPSILON 0x1.fffffep-1f // ~0.99999994
#define SBIAS 0.0008
#define PI 3.1415926535
#define INV_PI 0.3183098861
#define LUT_SIZE 16
#define MIN_NORMAL_INT 0.33f


// TEXTURES & LUTs
#define SHEEN_LUT_INDEX 0
#define GGX_ESS_LUT_INDEX 1


// ___ GI Reuse ___
#define TEMP_MCAP_GI 30

#define SPAT_MCAP_GI 240
#define SPAT_EXP_GI 1.0f
#define SPAT_RAD_MAX_GI 32
#define SPAT_RAD_MIN_GI 24

#define SPAT_COUNT_MAX_GI 0
#define SPAT_COUNT_MIN_GI 0
#define SPAT_TRIS_GI 6

#define SPAT_MIN_M_GI 5
#define SPAT_BETA_GI 1.0f