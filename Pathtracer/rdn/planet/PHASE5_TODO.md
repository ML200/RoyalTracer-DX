# Planet streaming

The planet path is split into CPU selection, tessellation, GPU geometry pools,
BLAS streaming, and TLAS assembly. `StreamOrchestrator` owns the per-frame
pipeline; `GenerationBuilder` handles asynchronous terrain rebuilds.

## Current contracts

- Terrain vertices are written in a chunk-local frame and instances translate
  them by `anchor_world - scene_origin`.
- The unified TLAS contains scene instances, resident terrain chunks, and
  external instances. Newly built chunks join on the next frame.
- Terrain shading derives surface data from the hit position and the same
  heightmap parameters used by `tessellator.cpp`.

## Follow-up work

- Profile tessellation and reduce repeated heightmap samples.
- Add permanent fallback face geometry to cover initial streaming gaps.
- Add predictive prefetch and geomorphing when the streaming path is stable.
- Evaluate sparse heightmap storage and BLAS compaction for large worlds.

The code and tests in this directory are the source of truth for behavior;
this file only records the remaining engineering work.
