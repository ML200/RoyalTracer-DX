#ifndef DLSS_GUIDE_LAYOUT
#define DLSS_GUIDE_LAYOUT

// Guide depth encoding is shared by producers and consumers.
#define DLSS_GUIDE_DEPTH_NEAR 0.01f
#define DLSS_GUIDE_DEPTH_FAR 100000000.0f

#define DLSS_SPEC_HIT_MAX 65504.0f

// dbg_dlssLayer carries the debug view in its low byte and guide options above it.
#define DLSS_DBG_LAYER_MASK 0xFFu
#define DLSS_GUIDE_OPT_NO_PSR 0x100u
#define DLSS_GUIDE_OPT_NO_MV_BLEND 0x200u

// Primary surface replacement: the camera pass walks the primary reflection through delta
// surfaces and packs the chain's end into this scratch slice (virtual normal, albedo, chain
// throughput, packed roughness/metalness/flags/bounces) for the shading pass.
#define DLSS_PSR_PROBE_SLOT 5u
// Chain end position and instance for candidates; scratch slice 4 keeps the plain first reflection hit.
#define DLSS_PSR_CHAIN_SLOT 7u
#define DLSS_PSR_FLAG_VALID 1u
#define DLSS_PSR_FLAG_EMITTER 2u
#define DLSS_PSR_FLAG_SKY 4u
#define DLSS_PSR_FLAG_DELTA 8u
#define DLSS_PSR_FLAG_GLASS 16u
// Segments a delta chain may follow beyond the primary hit.
#define DLSS_PSR_MAX_CHAIN 4u

// Only delta lobes are replaced: full below ROUGH_FULL, none from ROUGH_END on, which is the
// roughness below which the path tracer samples a lobe as a perfect mirror.
#define DLSS_PSR_ROUGH_FULL 0.04f
#define DLSS_PSR_ROUGH_END 0.06f
// Every merged guide follows the mirror's Fresnel share of the pixel's reflectance. For diffuse
// motion, 0 hands over to the virtual surface once that share passes half, 1 blends in proportion.
#define DLSS_PSR_MV_MIX 1.0f
#endif
