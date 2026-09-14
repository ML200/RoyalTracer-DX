#ifndef SKYBAKE_LAYOUT_H
#define SKYBAKE_LAYOUT_H

// Sky-bake offsets are shared by the host and shader passes.
#define SKYBAKE_SUN_OFFSET 0u
#define SKYBAKE_SUN_BYTES 64u
#define SKYBAKE_LUT_OFFSET 256u
#define SKYBAKE_LUT_W 256u
#define SKYBAKE_LUT_H 128u
#define SKYBAKE_LUT_TEXEL_BYTES 16u
#define SKYBAKE_LUT_BYTES (SKYBAKE_LUT_W * SKYBAKE_LUT_H * SKYBAKE_LUT_TEXEL_BYTES)
#define SKYBAKE_BYTES (SKYBAKE_LUT_OFFSET + SKYBAKE_LUT_BYTES)

#define SKYBAKE_THREADS 64u
#define SKYBAKE_GROUPS ((SKYBAKE_LUT_W * SKYBAKE_LUT_H) / SKYBAKE_THREADS)
#ifdef __cplusplus
static_assert(SKYBAKE_GROUPS == 512u, "keep the Pass_pt_skybake_v8 fx: token in Renderer.cpp in sync");
static_assert(SKYBAKE_SUN_OFFSET + SKYBAKE_SUN_BYTES <= SKYBAKE_LUT_OFFSET);
#endif
#endif
