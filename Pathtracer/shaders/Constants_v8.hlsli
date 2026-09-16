#define EPSILON 0.000003
#define ONE_MINUS_EPSILON 0x1.fffffep-1f
#define SBIAS 0.0008
#define PI 3.1415926535
#define INV_PI 0.3183098861
#define LUT_SIZE 16
#define MIN_NORMAL_INT 0.33f

#define SMOOTH_SPECULAR_THRESHOLD 0.06f

#define BROAD_GGX_ROUGHNESS 0.8f
#define kInvalidPixel -1u

#ifndef SHADING_DEBUG_SLICES
#define SHADING_DEBUG_SLICES 1
#endif

#include "DlssGuideLayout.h"

#define RAY_TMAX_PLANET 1e9f

#define ATM_DEBUG_RING 0

#define SHEEN_LUT_INDEX 0
#define GGX_ESS_LUT_INDEX 1

#define TEMP_MCAP 8

#define SPAT_MCAP 100000
#define SPAT_EXP  0.8f
#define SPAT_RAD_MAX 48
#define SPAT_RAD_MIN 8

#define SPAT_COUNT_MAX 3

#define MATID_ENV_MISS    0xFFFFFFFFu
#define MATID_LIGHT_TRI   0xFFFFFFFEu
#define IsSentinelMatID(mid) ((mid) >= MATID_LIGHT_TRI)

#define MATID_SSS_VOLUME_BIT 0x40000000u
#define MATID_SSS_EXIT_BIT   0x20000000u
#define IsVolumeVertex(mid)  (((mid) & MATID_SSS_VOLUME_BIT) != 0u && !IsSentinelMatID(mid))
#define IsSSSExitVertex(mid) (((mid) & MATID_SSS_EXIT_BIT)   != 0u && !IsSentinelMatID(mid))
#define MatIDBase(mid)       ((mid) & ~(MATID_SSS_VOLUME_BIT | MATID_SSS_EXIT_BIT))

#define REUSE_ROUGHNESS_MIN 0.15f
#define REUSE_ROUGHNESS_MAX 0.6f

#define SPMIS_GLASS_ROUGHNESS_MIN 0.3f

#define BOIL_STRENGTH_TEMP 0.2f
#define BOIL_MIN_AVG_TEMP  1e-8f
