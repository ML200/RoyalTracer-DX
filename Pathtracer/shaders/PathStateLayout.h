#pragma once

// Per-pixel path-state buffer, stored [plane][pixel]: every value below is a byte offset per pixel,
// shared with the host through PS_PATH_STATE_BYTES. The training lanes are at most a quarter of the
// pixels (training stride >= 2), so their per-lane records fit in the planes at the end.
#define PS_LITE_PARK_PLANE   0u      // diffuse reuse: parked candidate point
#define PS_LITE_PARK_BYTES   16u
#define PS_LITE_STATE_PLANE  (PS_LITE_PARK_PLANE + PS_LITE_PARK_BYTES)   // diffuse reuse: candidate state
#define PS_LITE_STATE_BYTES  16u
#define PS_PRIMARY_PLANE     (PS_LITE_STATE_PLANE + PS_LITE_STATE_BYTES) // primary IOR, medium, absorption
#define PS_PRIMARY_BYTES     16u

// Deferred shading records of the split path tracer (PtDefer_v8.hlsli).
#define DV_VERT_PLANE        (PS_PRIMARY_PLANE + PS_PRIMARY_BYTES)
#define DV_VERT_BYTES        48u
#define DV_SCATTER_PLANE     (DV_VERT_PLANE + DV_VERT_BYTES)
#define DV_SCATTER_BYTES     16u
#define DV_PATH_PLANE        (DV_SCATTER_PLANE + DV_SCATTER_BYTES)
#define DV_PATH_BYTES        32u
#define DV_LITE_PLANE        (DV_PATH_PLANE + DV_PATH_BYTES)
#define DV_LITE_BYTES        4u
#define DV_END_PLANE         (DV_LITE_PLANE + DV_LITE_BYTES)
#define DV_END_BYTES         24u
#define DV_LIGHT_PLANE       (DV_END_PLANE + DV_END_BYTES)
#define DV_LIGHT_BYTES       60u
#define PS_DEFER_END         (DV_LIGHT_PLANE + DV_LIGHT_BYTES)

// Diffuse-reuse shift scratch (three 12-byte slots, then mask and own target): the shift pass
// writes it after the sample loop, so it aliases the vertex record, which is dead by then. It must
// stay clear of the park, state and primary planes, which the merge pass still reads.
#define PS_LITE_SHIFT_PLANE  DV_VERT_PLANE
#define PS_LITE_SHIFT_BYTES  DV_VERT_BYTES

// Training lanes: two guide roots (32 bytes each) and the 48-byte propagation state per lane.
#define PS_GUIDE_ROOTS_PLANE PS_DEFER_END
#define PS_GUIDE_ROOTS_BYTES 16u
#define PS_TRAIN_SPILL_PLANE (PS_GUIDE_ROOTS_PLANE + PS_GUIDE_ROOTS_BYTES)
#define PS_TRAIN_SPILL_BYTES 12u
#define PS_PATH_STATE_BYTES  (PS_TRAIN_SPILL_PLANE + PS_TRAIN_SPILL_BYTES)

#ifdef __cplusplus
static_assert(PS_DEFER_END == 232u, "deferred record layout changed; update the shader planes");
static_assert(PS_PATH_STATE_BYTES == 260u, "path-state size changed; update the shader planes");
static_assert(PS_LITE_SHIFT_BYTES >= 44u, "the lite shift record is 44 bytes per pixel; keep it inside the aliased vertex record");
#endif
