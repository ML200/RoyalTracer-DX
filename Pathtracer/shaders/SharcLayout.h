#ifndef SHARC_LAYOUT_H
#define SHARC_LAYOUT_H
// Shared by C++ and HLSL. One persistent allocation, independent of resolution.
// State words (empty / locked / key hash) live contiguously at the front in
// bucket order, so probing a 16-slot bucket is one 64-byte read. Entries follow.
#define SHARC_CAPACITY (1u << 20u)
#define SHARC_ENTRY_BYTES 160u
#define SHARC_BUCKET_SIZE 16u
#define SHARC_STATE_BYTES (SHARC_CAPACITY * 4u)
#define SHARC_BUFFER_BYTES (SHARC_STATE_BYTES + SHARC_CAPACITY * SHARC_ENTRY_BYTES)
#define SHARC_MAX_LEVEL 24u
#define SHARC_GROUP_SIZE 256u
#define SHARC_ROOT_CONSTANTS 56u
// Packed into sharc_enabled; zero still disables all cache work.
#define SHARC_DEBUG_MODE_SHIFT 1u
#define SHARC_DEBUG_MODE_MASK 3u
#define SHARC_DEBUG_CELLS 1u
#define SHARC_DEBUG_LIGHTING 2u
#define SHARC_DEBUG_OTHER_LEVEL_BIT 8u
#define SHARC_DEBUG_SCRATCH 14u
#ifdef __cplusplus
static_assert((SHARC_CAPACITY & (SHARC_CAPACITY - 1u)) == 0u);
static_assert(SHARC_CAPACITY % SHARC_BUCKET_SIZE == 0u);
static_assert(SHARC_CAPACITY % SHARC_GROUP_SIZE == 0u);
static_assert(SHARC_ROOT_CONSTANTS + 7u <= 64u);
#endif
#endif
