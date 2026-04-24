//====================================================================
//NUMERIC CONSTANTS
//====================================================================
#define EPSILON 0.00003
#define ONE_MINUS_EPSILON 0x1.fffffep-1f //~0.99999994
#define SBIAS 0.0008
#define PI 3.1415926535
#define INV_PI 0.3183098861
#define LUT_SIZE 16
#define MIN_NORMAL_INT 0.33f
#define SMOOTH_SPECULAR_THRESHOLD 0.06f
#define kInvalidPixel -1u


//====================================================================
//TEXTURE AND LUT INDICES
//====================================================================
#define SHEEN_LUT_INDEX 0
#define GGX_ESS_LUT_INDEX 1

//====================================================================
//REUSE CAPS
//====================================================================
#define TEMP_MCAP 8

#define SPAT_MCAP 48
#define SPAT_EXP  0.8f
#define SPAT_RAD_MAX 48
#define SPAT_RAD_MIN 8

#define SPAT_COUNT_MAX 3

//====================================================================
//UNIFIED DI+GI RESERVOIR, MATID DISCRIMINATOR
//====================================================================
//Sentinel matIDs flag direct-lighting candidates inside the unified reservoir.
//Reconnect branches on these values to pick the correct reconnection formula,
//all x2 BSDF / material fetches are skipped for sentinel samples.
//MATID_ENV_MISS:    x2 stores a DIRECTION, not a position. No geometry
//                   term, no x2 BSDF. Jn = 1.
//MATID_LIGHT_TRI:   x2 is a world position on an emissive triangle.
//                   n2_s is the light's surface normal. No x2 BSDF,
//                   L2 carries the triangle's emission directly.
//MATID_NRC_VLIGHT:  x2 is a real surface position (depth-1 vertex), but
//                   L2 is the NRC cache prediction at x2 — i.e. we
//                   treat the vertex AS a virtual light source. Same
//                   reconnect math as MATID_LIGHT_TRI (geometric Jn,
//                   no F2 BSDF eval). Used only when nrc::Settings::
//                   aggressiveCacheTerm is on. Adds a small directional
//                   bias under shift (cache prediction was for the
//                   original x1's view direction, not the shifted one),
//                   accepted in exchange for skipping the F2 cost.
//Real triangle materials occupy IDs < MATID_NRC_VLIGHT.
#define MATID_ENV_MISS    0xFFFFFFFFu
#define MATID_LIGHT_TRI   0xFFFFFFFEu
#define MATID_NRC_VLIGHT  0xFFFFFFFDu
#define IsSentinelMatID(mid) ((mid) >= MATID_NRC_VLIGHT)

//====================================================================
//ROUGHNESS REUSE GATE
//====================================================================
#define REUSE_ROUGHNESS_MIN 0.15f
#define REUSE_ROUGHNESS_MAX 0.6f

//====================================================================
//BOILING FILTER
//====================================================================
//Lower strength means less aggressive clamping.
#define BOIL_STRENGTH_TEMP 0.2f
#define BOIL_MIN_AVG_TEMP  1e-8f
