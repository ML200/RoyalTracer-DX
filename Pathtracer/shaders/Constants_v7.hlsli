#define EPSILON 0.000006
#define PI 3.1415926535
#define LUT_SIZE 16

// ___ DI initial sampling ___
#define NEE_SAMPLES_DI 8
#define BSDF_SAMPLES_DI 1

#define WAVE_CANDIDATES_DI 8

// ___ GI initial sampling ___
#define NEE_SAMPLES_GI 4
#define BSDF_SAMPLES_GI 3

// ___ DI Reuse ___
#define TEMP_MCAP_DI 30

#define SPAT_MCAP_DI 30
#define SPAT_EXP_DI 1.0f
#define SPAT_RAD 20

#define SPAT_COUNT_MAX_DI 5
#define SPAT_COUNT_MIN_DI 1
#define SPAT_TRIS_DI 3
#define SPAT_WAVE_CANDIDATES_DI 32