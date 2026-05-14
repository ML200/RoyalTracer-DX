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
