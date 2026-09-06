#ifndef SHARC_LAYOUT_H
#define SHARC_LAYOUT_H
// Shared by C++ and HLSL. One persistent allocation, independent of resolution.
// State words (empty / locked / key hash) live contiguously at the front in
// bucket order, so probing a 16-slot bucket is one 64-byte read. Entries follow.
#define SHARC_CAPACITY (1u << 20u)
#define SHARC_ENTRY_BYTES 160u
#define SHARC_BUCKET_SIZE 16u
#define SHARC_STATE_BYTES (SHARC_CAPACITY * 4u)
#define SHARC_CACHE_BYTES (SHARC_STATE_BYTES + SHARC_CAPACITY * SHARC_ENTRY_BYTES)
// Path-guiding receiver table (SharcGuide_v8.hlsli), appended to the same
// allocation so it needs no root parameter of its own. Same bucketed state
// word scheme; entries hold 8 bright-patch candidates plus an irradiance mean.
#define GUIDE_CAPACITY (1u << 17u)
#define GUIDE_ENTRY_BYTES 160u
#define GUIDE_SLOTS 8u
#define GUIDE_STATE_BYTES (GUIDE_CAPACITY * 4u)
#define GUIDE_BYTES (GUIDE_STATE_BYTES + GUIDE_CAPACITY * GUIDE_ENTRY_BYTES)
#define SHARC_BUFFER_BYTES (SHARC_CACHE_BYTES + GUIDE_BYTES)
#define SHARC_MAX_LEVEL 24u
#define SHARC_GROUP_SIZE 256u
#define SHARC_ROOT_CONSTANTS 57u
// Packed into sharc_enabled; zero still disables all cache work.
#define SHARC_DEBUG_MODE_SHIFT 1u
#define SHARC_DEBUG_MODE_MASK 3u
#define SHARC_DEBUG_CELLS 1u
#define SHARC_DEBUG_LIGHTING 2u
#define SHARC_DEBUG_GUIDING 3u
#define SHARC_DEBUG_OTHER_LEVEL_BIT 8u
#define SHARC_DEBUG_SCRATCH 14u
// guide_params root constant (slot 56). Bit 0 enables guiding, bits 1-8 hold
// the guided-fraction cap x255, bits 9-11 the receiver level offset, bits
// 12-19 the candidate lifetime / 8 frames, bit 20 guides training paths,
// bits 21-27 hold the patch radius x32 in cell widths and bits 28-30 the
// deepest path vertex that is guided and trained (1 = primary only).
#define GUIDE_PARAM_ENABLED 1u
#define GUIDE_PARAM_QMAX_SHIFT 1u
#define GUIDE_PARAM_LEVEL_SHIFT 9u
#define GUIDE_PARAM_LIFETIME_SHIFT 12u
#define GUIDE_PARAM_TRAIN (1u << 20u)
#define GUIDE_PARAM_RADIUS_SHIFT 21u
#define GUIDE_PARAM_DEPTH_SHIFT 28u
#ifdef __cplusplus
static_assert((SHARC_CAPACITY & (SHARC_CAPACITY - 1u)) == 0u);
static_assert(SHARC_CAPACITY % SHARC_BUCKET_SIZE == 0u);
static_assert(SHARC_CAPACITY % SHARC_GROUP_SIZE == 0u);
static_assert((GUIDE_CAPACITY & (GUIDE_CAPACITY - 1u)) == 0u);
static_assert(GUIDE_CAPACITY % SHARC_BUCKET_SIZE == 0u);
static_assert(GUIDE_CAPACITY <= SHARC_CAPACITY); // the prepare dispatch covers both
static_assert(SHARC_ROOT_CONSTANTS + 7u <= 64u);
#endif
#endif
