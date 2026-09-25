#pragma once

// Sky-bake offsets are shared by the host and shader passes.
#define SKYBAKE_SUN_OFFSET 0u
#define SKYBAKE_SUN_BYTES 64u
#define SKYBAKE_LUT_OFFSET 256u
#define SKYBAKE_LUT_W 256u
#define SKYBAKE_LUT_H 128u
#define SKYBAKE_LUT_TEXEL_BYTES 16u
#define SKYBAKE_LUT_BYTES (SKYBAKE_LUT_W * SKYBAKE_LUT_H * SKYBAKE_LUT_TEXEL_BYTES)
#define SKYBAKE_BYTES (SKYBAKE_LUT_OFFSET + SKYBAKE_LUT_BYTES)

// The sky's irradiance on a horizontal plane (the sun excluded), summed over the bake's upper half
// as it is baked: three 64-bit fixed-point sums, in one of two slots by frame parity so the slot a
// frame sums into was cleared the frame before, while nothing read or wrote it.
#define SKYBAKE_IRRADIANCE_OFFSET 64u
#define SKYBAKE_IRRADIANCE_SLOT_BYTES 32u
#define SKYBAKE_IRRADIANCE_SCALE 1048576.0f

#define SKYBAKE_THREADS 64u
#define SKYBAKE_GROUPS ((SKYBAKE_LUT_W * SKYBAKE_LUT_H) / SKYBAKE_THREADS)
#ifdef __cplusplus
static_assert(SKYBAKE_GROUPS == 512u, "keep the Pass_pt_skybake_v8 fx: token in Renderer.cpp in sync");
static_assert(SKYBAKE_SUN_OFFSET + SKYBAKE_SUN_BYTES <= SKYBAKE_IRRADIANCE_OFFSET);
static_assert(SKYBAKE_IRRADIANCE_OFFSET + 2u * SKYBAKE_IRRADIANCE_SLOT_BYTES <= SKYBAKE_LUT_OFFSET);
#endif
