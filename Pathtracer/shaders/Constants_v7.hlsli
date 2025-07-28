#define EPSILON 0.000003
#define PI 3.1415926535
#define LUT_SIZE 16

// ___ DI initial sampling ___
#define NEE_SAMPLES_DI 4
#define BSDF_SAMPLES_DI 1

// ___ GI initial sampling ___
#define NEE_SAMPLES_GI 4
#define BSDF_SAMPLES_GI 2

// ___ DI Reuse ___
#define TEMP_MCAP_DI 30

#define SPAT_MCAP_DI 500
#define SPAT_EXP_DI 1.0f
#define SPAT_RAD_MAX 32
#define SPAT_RAD_MIN 24

#define SPAT_COUNT_MAX_DI 2
#define SPAT_COUNT_MIN_DI 1
#define SPAT_TRIS_DI 1


// ___ GI Reuse ___
#define TEMP_MCAP_GI 30

#define SPAT_MCAP_GI 120
#define SPAT_EXP_GI 0.9f
#define SPAT_RAD_MAX_GI 32
#define SPAT_RAD_MIN_GI 32

#define SPAT_COUNT_MAX_GI 2
#define SPAT_COUNT_MIN_GI 1
#define SPAT_TRIS_GI 1

#define SPAT_MIN_M_GI 5
#define SPAT_BETA_GI 1.0f


// Denoiser settings spatial
/*#define PLANE_DISTANCE_THRESHOLD 0.02f
#define NORMAL_POWER 16.0f
#define ALBEDO_SIGMA 0.2f
#define SIGMA_SMOOTH   1.0f
#define SIGMA_ROUGH    4.0f
#define ROUGH_DIFF_SIGMA 0.15f
#define ILLUM_SIGMA 0.2f
#define COLOR_RANGE_REL  .07f

// Denoiser settings temporal
#define ROUGHNESS_DECAY 0.05f*/