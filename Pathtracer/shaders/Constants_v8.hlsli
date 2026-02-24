#define EPSILON 0.000003
#define ONE_MINUS_EPSILON 0x1.fffffep-1f // ~0.99999994
#define SBIAS 0.0008
#define PI 3.1415926535
#define INV_PI 0.3183098861
#define LUT_SIZE 16
#define MIN_NORMAL_INT 0.33f
#define kInvalidPixel -1u


// TEXTURES & LUTs
#define SHEEN_LUT_INDEX 0
#define GGX_ESS_LUT_INDEX 1

// ___ DI Reuse ___
#define TEMP_MCAP_DI 24

#define SPAT_MCAP_DI 48
#define SPAT_EXP_DI 1.0f
#define SPAT_RAD_MAX 32
#define SPAT_RAD_MIN 24

#define SPAT_COUNT_MAX_DI 2
#define SPAT_COUNT_MIN_DI 1
#define SPAT_TRIS_DI 6


// ___ GI Reuse ___
#define TEMP_MCAP_GI 24

#define SPAT_MCAP_GI 48
#define SPAT_EXP_GI 1.0f
#define SPAT_RAD_MAX_GI 32
#define SPAT_RAD_MIN_GI 24

#define SPAT_COUNT_MAX_GI 2
#define SPAT_COUNT_MIN_GI 1
#define SPAT_TRIS_GI 3

#define SPAT_MIN_M_GI 5
#define SPAT_BETA_GI 3.0f


// Denoiser settings spatial
#define PLANE_DISTANCE_THRESHOLD 0.02f
#define NORMAL_POWER 16.0f
#define ALBEDO_SIGMA 0.2f
#define SIGMA_SMOOTH   1.0f
#define SIGMA_ROUGH    4.0f
#define ROUGH_DIFF_SIGMA 0.15f
#define ILLUM_SIGMA 0.2f
#define COLOR_RANGE_REL  .07f

// Denoiser settings temporal
#define ROUGHNESS_DECAY 0.05f