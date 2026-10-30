#pragma once

// Guide depth encoding, shared by producers and consumers.
#define DLSS_GUIDE_DEPTH_NEAR 0.01f
#define DLSS_GUIDE_DEPTH_FAR 100000000.0f

#define DLSS_SPEC_HIT_MAX 65504.0f

// dbg_dlssLayer: debug view in the low byte, guide options above, the virtual geometry view in bits 24-27.
#define DLSS_DBG_LAYER_MASK 0xFFu
#define DLSS_GUIDE_OPT_NO_PSR 0x100u
#define DLSS_GUIDE_OPT_NO_MV_BLEND 0x200u

#define GEOMETRY_VIEW_SHIFT 24u
#define GEOMETRY_VIEW_MASK 0xFu
#define GEOMETRY_VIEW_OFF 0u
#define GEOMETRY_VIEW_CLUSTERS 1u
#define GEOMETRY_VIEW_LOD_LEVEL 2u
#define GEOMETRY_VIEW_TRIANGLES 3u
#define GEOMETRY_VIEW_TRIANGLE_SIZE 4u
#define GEOMETRY_VIEW_INSTANCES 5u
#define GEOMETRY_VIEW_RESIDENCY 6u
#define GEOMETRY_VIEW_COUNT 7u
// Scratch slice the camera pass writes the view's colour into, at render resolution.
#define GEOMETRY_VIEW_SCRATCH 14u

// ClusterRecord.info: the cluster's hierarchy level (0: full detail) and what kind of cluster it is.
#define CLUSTER_INFO_LEVEL_MASK 0xFFu
#define CLUSTER_INFO_KIND_SHIFT 8u
#define CLUSTER_KIND_STATIC 0u   // geometry without a hierarchy
#define CLUSTER_KIND_EXCLUDED 1u // in a mesh with a hierarchy, kept out of it (alpha, emissive, small regions)
#define CLUSTER_KIND_RESIDENT 2u // hierarchy, every level in memory
#define CLUSTER_KIND_ROOT 3u     // hierarchy, a root page (always resident)
#define CLUSTER_KIND_STREAMED 4u // hierarchy, a streamed page

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
