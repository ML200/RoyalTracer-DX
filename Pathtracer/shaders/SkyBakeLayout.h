#ifndef SKYBAKE_LAYOUT_H
#define SKYBAKE_LAYOUT_H
// Shared by C++ and HLSL. Per-frame sky bake of the regular path tracer
// (Pass_pt_skybake_v8.hlsl -> SkyBake_v8.hlsli), parked at the front of the
// SPMIS hash-grid buffer (root UAV u25), which is idle while the path tracer
// owns the frame. The host keeps that buffer at least SKYBAKE_BYTES large.
#define SKYBAKE_SUN_OFFSET 0u
#define SKYBAKE_SUN_BYTES 64u
#define SKYBAKE_LUT_OFFSET 256u
#define SKYBAKE_LUT_W 256u
#define SKYBAKE_LUT_H 128u
#define SKYBAKE_LUT_TEXEL_BYTES 16u
#define SKYBAKE_LUT_BYTES (SKYBAKE_LUT_W * SKYBAKE_LUT_H * SKYBAKE_LUT_TEXEL_BYTES)
#define SKYBAKE_BYTES (SKYBAKE_LUT_OFFSET + SKYBAKE_LUT_BYTES)
// Bake dispatch: one thread per texel, numthreads(SKYBAKE_THREADS, 1, 1). The
// pass-list token carries the group count as a literal (fx:512).
#define SKYBAKE_THREADS 64u
#define SKYBAKE_GROUPS ((SKYBAKE_LUT_W * SKYBAKE_LUT_H) / SKYBAKE_THREADS)
#ifdef __cplusplus
static_assert(SKYBAKE_GROUPS == 512u, "keep the Pass_pt_skybake_v8 fx: token in Renderer.cpp in sync");
static_assert(SKYBAKE_SUN_OFFSET + SKYBAKE_SUN_BYTES <= SKYBAKE_LUT_OFFSET);
#endif
#endif
