#pragma once

// Guide depth encoding, shared by producers and consumers.
#define DLSS_GUIDE_DEPTH_NEAR 0.01f
#define DLSS_GUIDE_DEPTH_FAR 100000000.0f

#define DLSS_SPEC_HIT_MAX 65504.0f

// dbg_dlssLayer: debug view in the low byte, guide options above.
#define DLSS_DBG_LAYER_MASK 0xFFu
#define DLSS_GUIDE_OPT_NO_PSR 0x100u
#define DLSS_GUIDE_OPT_NO_MV_BLEND 0x200u

// Scratch slice of the packed PSR chain end (camera pass -> shading pass).
#define DLSS_PSR_PROBE_SLOT 5u
// Chain end position/instance; slice 4 keeps the first reflection hit.
#define DLSS_PSR_CHAIN_SLOT 7u
#define DLSS_PSR_FLAG_VALID 1u
#define DLSS_PSR_FLAG_EMITTER 2u
#define DLSS_PSR_FLAG_SKY 4u
#define DLSS_PSR_FLAG_DELTA 8u
#define DLSS_PSR_FLAG_GLASS 16u
// Chain segments beyond the primary hit.
#define DLSS_PSR_MAX_CHAIN 4u

// Full below ROUGH_FULL; ROUGH_END = SMOOTH_SPECULAR_THRESHOLD.
#define DLSS_PSR_ROUGH_FULL 0.04f
#define DLSS_PSR_ROUGH_END 0.06f
// Diffuse motion: 0 = hand-over at half share, 1 = proportional.
#define DLSS_PSR_MV_MIX 1.0f
