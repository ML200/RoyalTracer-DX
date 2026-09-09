//====================================
//NUMERIC CONSTANTS
//====================================
#define EPSILON 0.000003
#define ONE_MINUS_EPSILON 0x1.fffffep-1f
#define SBIAS 0.0008
#define PI 3.1415926535
#define INV_PI 0.3183098861
#define LUT_SIZE 16
#define MIN_NORMAL_INT 0.33f
//BSDF delta lobe gate, low so VNDF sampling stays unbiased
#define SMOOTH_SPECULAR_THRESHOLD 0.06f
//A GGX lobe at least this rough is BROAD: its outgoing radiance is close to
//view-independent, so the radiance cache, the lite reservoir and guiding
//treat it together with the diffuse lobe (LOBE_BROAD, BXDF_v8.hlsli).
#define BROAD_GGX_ROUGHNESS 0.8f
#define kInvalidPixel -1u

//Postprocess debug comparison slices: "noisy" (gOutput 0, scratch slot 1),
//"gt" running-average reference (gOutput 2, gPermanentData), "albedo" (gOutput 5).
//The shipped image is the DLSS "clean" slice (gOutput 1); these are dev-only.
//1 (default) = write them. Set to 0 (e.g. -D SHADING_DEBUG_SLICES=0 in a shipping
//build) to drop their full-res FP32 scratch + gPermanentData traffic in
//Pass_shading / Pass_postprocess — a large share of shading's VRAM throughput.
#ifndef SHADING_DEBUG_SLICES
#define SHADING_DEBUG_SLICES 1
#endif

//====================================
//DLSS GUIDE DEPTH RANGE
//====================================
// R32F reverse-Z device depth, matching Streamline's guide projection. A 10 km
// far plane clipped horizon clouds to the sky sentinel; the shared range covers
// planetary views. Linear R16F specular hit distances have their own finite cap.
#include "DlssGuideLayout.h"

//====================================
//RAY TMAX
//====================================
//Planet wide ray range. Earth diameter is ~1.27e7 m; this leaves headroom
//for tangent grazing rays from orbital altitudes and for second bounces
//that escape upward into space before being marked as misses. Used for
//primary, reflection, and bounce rays in Pass_raygen_v8.hlsl. The sky
//depth fallback in Pass_shading_v8.hlsl reads cameraFar (CPU side),
//which should stay >= this value or distant surface hits appear closer
//than the sky in the DLSS RR depth buffer.
#define RAY_TMAX_PLANET 1e9f

//Atmosphere diagnostics: 0 = normal, 1 = transmittance/scatter, 2 = path
//contributions, 3 = sky disabled, 4 = surface normals.
#define ATM_DEBUG_RING 0


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

//SSS reconnection-vertex provenance bits, OR-ed into the (real) SSS material id of a
//reservoir's reconnection vertex. Real ids are < 2^29, so these stay below the sentinels
//and still route through the GI/surface branch. Mask off with MatIDBase before any
//g_mat[]/Load*() indexing.
//  VOLUME: x2 is the first in-medium scatter (Reconnect volume branch: phase + 1/dist^2).
//  EXIT:   x2 is a no-scatter pass-through exit surface; x1's coupling is the 1/pi diffuse
//          entry term (Reconnect surface branch overrides F1) — it is NOT a volume vertex.
#define MATID_SSS_VOLUME_BIT 0x40000000u
#define MATID_SSS_EXIT_BIT   0x20000000u
#define IsVolumeVertex(mid)  (((mid) & MATID_SSS_VOLUME_BIT) != 0u && !IsSentinelMatID(mid))
#define IsSSSExitVertex(mid) (((mid) & MATID_SSS_EXIT_BIT)   != 0u && !IsSentinelMatID(mid))
#define MatIDBase(mid)       ((mid) & ~(MATID_SSS_VOLUME_BIT | MATID_SSS_EXIT_BIT))

//====================================
//ROUGHNESS REUSE GATE
//====================================
#define REUSE_ROUGHNESS_MIN 0.15f
#define REUSE_ROUGHNESS_MAX 0.6f

//SPMIS spatial reuse is invalid on near-specular transmissive surfaces (glass): the tight
//transmit/refract lobe is strongly view-dependent, so resampling across neighbouring pixels
//(different view directions) produces tight-lobe p_hat spikes -> fireflies. Glass whose PRIMARY
//surface roughness is below this skips spatial reuse and keeps its (view-stable) temporal +
//canonical reservoir. Opaque specular is self-gating (mismatched candidates get p_hat=0).
#define SPMIS_GLASS_ROUGHNESS_MIN 0.3f

//====================================
//BOILING FILTER
//====================================
//lower strength clamps less
#define BOIL_STRENGTH_TEMP 0.2f
#define BOIL_MIN_AVG_TEMP  1e-8f
