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
#define SMOOTH_SPECULAR_THRESHOLD 0.06f
#define kInvalidPixel -1u


//====================================
//TEXTURE AND LUT INDICES
//====================================
#define SHEEN_LUT_INDEX 0
#define GGX_ESS_LUT_INDEX 1

//====================================
//REUSE CAPS
//====================================
#define TEMP_MCAP 8

#define SPAT_MCAP 48
#define SPAT_EXP  0.8f
#define SPAT_RAD_MAX 48
#define SPAT_RAD_MIN 8

#define SPAT_COUNT_MAX 3

//====================================
//UNIFIED DI+GI RESERVOIR MATID DISCRIMINATOR
//====================================
//sentinels flag DI candidates, Reconnect branches on value
//MATID_ENV_MISS, x2 is direction, Jn=1, no geom term, no x2 BSDF
//MATID_LIGHT_TRI, x2 is emissive triangle position, no x2 BSDF, L2 is emission
//real materials occupy IDs < MATID_LIGHT_TRI
#define MATID_ENV_MISS    0xFFFFFFFFu
#define MATID_LIGHT_TRI   0xFFFFFFFEu
#define IsSentinelMatID(mid) ((mid) >= MATID_LIGHT_TRI)

//====================================
//ROUGHNESS REUSE GATE
//====================================
#define REUSE_ROUGHNESS_MIN 0.15f
#define REUSE_ROUGHNESS_MAX 0.6f

//====================================
//BOILING FILTER
//====================================
//lower strength means less clamping
#define BOIL_STRENGTH_TEMP 0.2f
#define BOIL_MIN_AVG_TEMP  1e-8f
