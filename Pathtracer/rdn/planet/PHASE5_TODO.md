# Planet Streaming — Phase 5 Remaining Work

Handoff for continuing in a fresh session. Phases 0–4 of the DXR planet streaming
system are **complete and verified** (the `Pathtracer` engine builds and runs).
Phase 5 (TLAS integration + terrain shading) is **partly done**. This document
states exactly what is left.

## Orientation
- All planet code: `rdn/planet/`. The Phase 0 audit + renderer integration points
  are in `rdn/planet/INTEGRATION_NOTES.md` — read it first.
- System shape: cube-sphere quadtree -> CPU multicore tessellation -> per-frame
  BLAS streaming on an async compute queue -> (Phase 5) one unified TLAS.
- Test target: a cube-sphere with a procedural sin/cos heightmap (`HeightmapProcedural`).
- The `StreamOrchestrator` (`stream_orchestrator.{h,cpp}`) is a `Renderer` member
  (`m_planet`), already wired into `Renderer::RenderFrame` as
  `begin_frame` / `submit_work` / `end_frame`. It streams chunk BLASes today.
  **Nothing is visible yet** — there is no TLAS referencing the planet BLASes.
  Phase 5 adds that.

## Planet module files (rdn/planet/)
- Phase 1 (CPU chunk system): `coordinate_system`, `cube_sphere`, `chunk`, `chunk_manager` (.h/.cpp), `planet_tests.cpp`
- Phase 2 (GPU pools): `chunk_mesh.h`, `blas_pool`, `scratch_pool`, `upload_ring` (.h/.cpp)
- Phase 3 (tessellation): `heightmap_source`, `heightmap_procedural`, `worker_pool`, `tessellator` (.h/.cpp)
- Phase 4 (BVH stream pipeline): `queue_sync`, `build_queue`, `geometry_pool`, `stream_orchestrator` (.h/.cpp)
- Phase 5: `tlas_builder` (.h/.cpp) — **done**

## Phase 5 decisions (already made — do NOT re-ask the user)
- **TLAS**: ONE unified TLAS (scene meshes + terrain chunks + fallback), full
  rebuild every frame, built on the async COMPUTE queue.
- **Fallback layer**: 6 permanent low-LOD BLASes (one per cube face, lod 0,
  slightly inset radius) — always in the TLAS so the planet has no holes while
  chunks stream in.
- **Terrain shading**: full material, done PROCEDURALLY. The geometry pool is
  cycling (a chunk's vertex/index buffers are freed once its BLAS is built), so
  there is NO per-vertex data at shade time. Terrain shading re-derives
  normal/uv/material from the ray-hit world position (cube-sphere mapping + the
  heightmap formula). `ChunkVertex`'s oct-normal/uv fields are unused by shading.

## Done in Phase 5
- `rdn/planet/tlas_builder.{h,cpp}` — `TlasBuilder`: `init` / `begin` /
  `add_instance` / `build(compute_cl)`. Builds the unified TLAS on a compute
  command list, preallocated result/scratch, persistently-mapped instance
  descriptors. Compiles, in the `Pathtracer` target, no callers yet.

---

## 5a — TLAS INTEGRATION (C++ / D3D12)

### A1. 6-face fallback layer — in `StreamOrchestrator::init`
- Tessellate 6 chunks: `QuadNode{face, lod=0, x=0, y=0}` for face 0..5, via
  `tessellate_chunk`, at a slightly **inset** radius (e.g. `planet.radius` minus a
  small inset) so streamed chunks sit on top. Init-time, synchronous (a stall is fine).
- Build their 6 BLASes synchronously at init: claim 6 BLAS slots (kept
  **permanently** — never released), record copy + BLAS build on the planet
  compute/copy lists, submit, CPU-wait the compute fence.
- Store the 6 fallback `{blas_slot, anchor_world}` permanently in the orchestrator.
- This needs a small init-time synchronous build path, separate from the
  per-frame async pipeline.

### A2. Orchestrator owns a `TlasBuilder`
- Add `TlasBuilder m_tlas;` member; `m_tlas.init(device, max_instances)` in `init()`.
- `max_instances` = `MAX_BLAS_SLOTS` (512 terrain) + 6 (fallback) + scene-instance
  allowance (pass the scene's max from the renderer, or use a generous constant).

### A3. Per-frame unified TLAS assembly + build
- New orchestrator method, e.g. `build_tlas(sceneInstances, originWorld, terrainHitGroupIndex, compute_cl)`.
  Call it from `submit_work`, AFTER recording the BLAS builds on the compute list,
  BEFORE `SubmitPlanetCompute` — so one compute submission does BLAS builds + TLAS
  build and one compute fence covers both.
- `m_tlas.begin()`, then `add_instance` for:
  - **Scene instances** (`Scene::TLASInstance` from `m_scene.tlasInstances`): their
    transforms are already `sceneOriginWorld`-relative — pass as-is. Use
    `sceneOriginWorld` as the unified TLAS origin.
  - **6 fallback instances**: transform = translate by `(fallback.anchor_world -
    originWorld)` (FP64 -> f32), identity rotation. `instanceID` in the terrain
    range. `hitGroupContribution` = `terrainHitGroupIndex`. flags = FORCE_OPAQUE.
  - **Every `Ready` terrain chunk** (`m_chunks->for_each_live`, state==Ready):
    transform = translate by `(chunk.anchor_world - originWorld)`. BLAS VA =
    `m_blasPool.gpu_address(chunk.blas_slot)`. `instanceID` in the terrain range.
    `hitGroupContribution` = `terrainHitGroupIndex`. flags = FORCE_OPAQUE.
- `m_tlas.build(compute_cl)`.
- Note: chunk geometry is already world-oriented chunk-local (tessellator did
  `local = world_pos - anchor`), so the instance transform is translation-only.
- The TLAS only references already-`Ready` chunks (prior-frame BLASes) + fallback
  + scene — all fully built. Chunks built this frame join the TLAS next frame.

### A4. `instanceID` encoding for terrain
- Terrain + fallback instances need `instanceID` >= a `TERRAIN_INSTANCE_BASE`
  constant (larger than any scene instance count, e.g. `1 << 20`). Scene instances
  keep 0..N. The shader branches on `instID >= TERRAIN_INSTANCE_BASE`. With
  procedural shading the shader only needs the "is terrain" flag — surface is
  derived from the hit position.

### A5. Renderer surgery (`Renderer.cpp` / `Renderer_Pipeline.cpp`)
- `Renderer::PopulateCommandList` — the `// TLAS update:` block (tagged
  `// PLANET_INTEGRATION`, ~line 1288): the 3-tier dirty TLAS logic is superseded.
  The renderer must still call `RebuildTLASInstanceList` when the scene is dirty
  and hand `m_scene.tlasInstances` to the orchestrator each frame; the orchestrator
  now builds the unified TLAS. Remove/bypass the renderer's own `CreateTopLevelAS`
  per-frame call.
- `SceneBVH` SRV at descriptor-heap **slot 2** (created `Renderer_Pipeline.cpp:704-710`,
  `D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE`; HLSL declared
  `SceneBVH : register(t0)` in `Includes_v8.hlsli:448`): point it at
  `m_planet`'s unified TLAS (`TlasBuilder::result()` GPU VA). The TLAS result
  buffer is preallocated once -> create this SRV once, after orchestrator init.
- Graphics queue must wait the planet compute fence before ray dispatch. The
  orchestrator's `submit_work` -> `SubmitPlanetCompute` returns the compute fence
  value; expose it. In `DeviceContext::ExecuteAndPresent`, BEFORE the graphics
  `ExecuteCommandLists`, add `cmdQueue->Wait(planetComputeFence.fence(), value)`.
- `CameraView` likely needs a `DVec3 scene_origin` field so the orchestrator has
  the TLAS origin; `Renderer::MakePlanetCamera()` already computes it from
  `Camera::getSceneOriginWorld()`.

---

## 5b — TERRAIN SHADING (HLSL + SBT)

### B1. `TerrainHitGroup` + SBT (`Renderer_Pipeline.cpp`)
- ~line 570: add `pipeline.AddHitGroup(L"TerrainHitGroup", L"ClosestHit");`
  (the closest-hit stays the stub — shading is in raygen).
- SBT build ~lines 1105-1122: the SBT is 2 entries per scene instance. Append one
  `TerrainHitGroup` entry after the scene entries; its index = `2 * sceneInstanceCount`.
  That index is the `terrainHitGroupIndex` passed to `build_tlas` (A3).

### B2. Planet constants in the camera cbuffer
- `Includes_v8.hlsli` `cbuffer CameraParams : register(b0)` (~lines 132-210): add
  `float3 planetCenter; float planetRadius; float terrainHeightAmplitude;
  float terrainHeightFrequency;` (mind 16-byte alignment).
- Fill them in `Camera::UploadGPUBuffer` (`Camera.cpp:143-207`, the `extra[]` /
  settings tail) — match the cbuffer byte layout exactly.

### B3. Terrain branch in `EvalSurfaceState` (`Inline_RT_v8.hlsli:349-512`)
- At the top: `if (instID >= TERRAIN_INSTANCE_BASE) { ... produce HitInfo; return; }`.
- `HitInfo` (struct at `Inline_RT_v8.hlsli:13-19`): produce `hitPos` (world hit
  position), `hitNormal` (heightmap surface normal at `normalize(hitPos -
  planetCenter)` — re-derive by sampling the heightmap formula at the hit dir + 4
  neighbours, finite-difference; same as the CPU tessellator), `uv` (cube-sphere uv
  of the hit dir), `backface=false`, `lightID=0xFFFFFFFF`.
- The GI passes (`Pass_temp_gi_v8`, `Pass_spat_gi_shift_v8`) funnel through
  `BuildVertex` -> `EvalSurfaceState`, so this one branch covers primary + GI.

### B4. Terrain material branch (`GetMatIDFast` / `RefetchMaterial`, `Inline_RT_v8.hlsli`)
- For terrain `instID`s, produce a terrain material: `Kd` (albedo), `Pr`
  (roughness), `Pm` (metalness=0). A fixed albedo+roughness is fine for the first
  cut; procedural-by-slope/altitude later.

### B5. Heightmap formula parity
- The HLSL heightmap used in B3 MUST match `HeightmapProcedural::sample`
  (`heightmap_procedural.cpp`): `sin(dir.x*freq)*cos(dir.z*freq)*amplitude`, with
  the freq/amplitude from the cbuffer (B2), so the shaded surface matches the
  tessellated geometry.

---

## Verify (Phase 5 done-criteria)
- Path tracer renders the cube-sphere with procedural bumps.
- Camera flythrough shows chunks streaming in (LOD pops at boundaries are fine).
- Fallback layer always visible underneath — no holes during streaming.
- Atmosphere + clouds still work (they are a separate screen-space pass,
  unaffected by the TLAS).
- Build target: `Pathtracer`. (`PlanetTests` is unaffected by Phase 5.)
- Nsight: BLAS + TLAS builds on the compute queue; total BLAS+TLAS < 1 ms.

## Known issues / deferred (not Phase 5 blockers)
- Tessellation throughput is ~4.4 chunks/ms (target 20) — the tessellator's
  5-heightmap-samples-per-vertex normal path. Optimize later (batch via
  `sample_grid`, or per-worker scratch). Async dispatch (Phase 4) means this is
  streaming latency, not a frame hitch.
- Phase 6 (deferred): predictive prefetch, geomorphing, skirts, real heightmap
  (sparse tiled resource), BLAS compaction, CLAS migration.
