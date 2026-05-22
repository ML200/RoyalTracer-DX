# Planet Streaming — Integration Notes

Phase 0 audit output (2026-05-21, `cleanup` branch). Reference for Phases 1–5.
All audited sites are tagged in-source with `// PLANET_INTEGRATION:` — grep for that
string to jump to the exact line (line numbers below drift as code is edited).

## Codebase layout

- C++ engine: `Pathtracer/rdn/`
- HLSL shaders: `Pathtracer/shaders/`
- DXR helpers: `rdn/nv_helpers_dx12/` — `TopLevelASGenerator`, `BottomLevelASGenerator`,
  `ShaderBindingTableGenerator`, `RaytracingPipelineGenerator`
- New planet code: `rdn/planet/`
- Device is `ID3D12Device10`, command list is `ID3D12GraphicsCommandList10` (modern;
  DXR + `CreateCommandQueue` for compute/copy are all available).

## Audit targets

### 1. Device / queues / frames-in-flight — `DeviceContext`
`rdn/Core/DeviceContext.{h,cpp}`. Single source of truth. Owns the `ID3D12Device10`,
**one** command queue of type `D3D12_COMMAND_LIST_TYPE_DIRECT` (`cmdQueue`), per-buffer
command allocators, swapchain, RTV/DSV. Queue created in `CreateDeviceAndSwapChain`.
Frames-in-flight = `bufferCount` (swapchain `BufferCount`, = `FRAME_COUNT`, can be bumped
by DLSS-G). **No async compute queue. No copy queue.**

### 2. Per-frame fence / sync — `DeviceContext`
Single `ID3D12Fence` (`fence`), rolling `nextFenceValue`, `fenceValues[MAX_BACK_BUFFERS]`,
`fenceEvent`. `WaitForPreviousFrame()` stalls the CPU on the latest submitted fence value
(heavily serialized — not deep N-frame pipelining). `BeginFrame()` resets the current
frame's allocator and reopens the list. `ExecuteAndPresent()` signals **after** Present
(DLSS-G's Present hook may submit extra work). Cross-fence GPU sync precedent already
exists: `CloseExecuteAndSignal()` / `WaitAndReopen()` do split-submission against an
external fence for CUDA interop — Phase 4's copy→compute→graphics handoff can mirror
that pattern, but with real separate queues.

### 3. TLAS build — `Renderer::PopulateCommandList` / `Renderer::CreateTopLevelAS`
`rdn/Renderer.cpp` (`PopulateCommandList`) and `rdn/Renderer_Pipeline.cpp`
(`CreateTopLevelAS`). **Not rebuilt every frame.** 3-tier, gated on `m_scene.tlasDirty`:
- full rebuild on structural change (new generator, reallocates buffers),
- periodic in-place rebuild every 120 frames (anti-degradation),
- partial `UpdateAndRefit` for transform-only changes.

`CreateTopLevelAS` uses `nv_helpers_dx12::TopLevelASGenerator`; allocates
`pScratch`/`pResult`/`pInstanceDesc` on full rebuild. `pResult` is created in
`D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE` and never transitioned (correct).
TLAS SRV sits at descriptor-heap index 2. All TLAS work runs on the DIRECT queue.

### 4. Atmosphere / clouds — NOT in the TLAS
Atmosphere + volumetric clouds are a **screen-space compute pass**,
`Pass_clouds_primary_v8.hlsl`, registered in the `m_passes.Build({...})` list in the
`Renderer` constructor. It marches unified atmosphere + cloud scatter and writes sky
colour + combined transmittance into scratch slots; `Pass_shading_v8.hlsl` composites.
The renderer is triangle-geometry-only (procedural-primitive / AABB support was removed).
There are **no procedural-primitive instances** for Phase 5 to append.

### 5. Closest-hit / opaque surfaces
`Hit_v8.hlsl::ClosestHit` is an **empty stub** (D3D12 requires the symbol; raygen uses
SER). Real surface shading happens in `Pass_raygen_v8.hlsl`'s HIT block:
`instID = hitObj.GetInstanceIndex()`, then material via `GetMatIDFast` /
`EvalSurfaceState` / `RefetchMaterial`. Hit groups (`Renderer_Pipeline.cpp`):
`OpaqueHitGroup` (ClosestHit) and `AlphaHitGroup` (ClosestHit + `AlphaTestAnyHit`); SBT
has 2 entries per instance (`hitGroupContribution = i*2`, set in
`Scene::RebuildTLASInstanceList`). The ReSTIR GI passes (`Pass_temp_gi_v8`,
`Pass_spat_gi_shift_v8`) trace the same TLAS + SBT, so terrain must shade correctly there
too.

## Infrastructure verification ("verify before continuing")

| Item | Status | Action |
|------|--------|--------|
| Async compute queue | **ABSENT** | Phase 4 prereq — add `computeQueue` + allocators/lists to `DeviceContext` (plan pre-authorizes "create one if not") |
| Copy queue | **ABSENT** | Phase 4 prereq — add `copyQueue` + allocators/lists to `DeviceContext` |
| Debug layer | Present, compile-time **OFF** (`DXDIAG_ENABLE_DEBUG_LAYER=0` in `Diagnostics.h`; off for 2–10× perf cost). DRED always on. | For bringup set the macro to 1 |
| GPU-based validation | **NOT wired** (no `SetEnableGPUBasedValidation`) | Add an `ID3D12Debug1::SetEnableGPUBasedValidation` call inside `EnableDebugLayerAndDred`, under the same macro |

## Deviations from the plan as written

**A. TLAS is not rebuilt every frame.** Phase 5 assumes an unconditional full rebuild.
Recommendation: for the unified TLAS (scene meshes + terrain chunks + fallback layer),
adopt full rebuild every frame over **max-sized** `pScratch`/`pResult`/`pInstanceDesc`
(Phase 2 pool philosophy — allocated once, never resized). The 3-tier dirty scheme
becomes redundant once terrain streams (the instance set changes most frames anyway).
A ~600-instance TLAS rebuild is well under the 1 ms budget on a 5090. *This changes
static-scene behaviour — confirm acceptable.*

**B. No atmosphere/cloud instances to append.** Phase 5's "append atmosphere/cloud
procedural-primitive instances" step is moot — they are screen-space. Terrain hits feed
the cloud composite like any other geometry; the only work is a Phase 5 visual check
that terrain interacts correctly with the atmosphere/cloud composite.

**C. The closest-hit shader is a stub.** The plan's "terrain material branch in the
closest-hit shader" actually belongs in `Pass_raygen_v8.hlsl`'s HIT block (and the GI
raygen passes), keyed on `instID` range. Two options for terrain material:
- (a) give terrain chunks real material-SoA entries → they flow through the existing
  `GetMatIDFast`/`EvalSurfaceState` path unchanged. **Blocker:** `EvalSurfaceState`
  reads the global scene vertex buffer in the scene `Vertex` layout; terrain chunk
  BLASes use the Phase-3 16-byte vertex format and their own slot buffers — that data
  is not in the global buffers.
- (b) an explicit `if (instID in terrainRange)` branch with a terrain-specific
  surface-eval that reads the terrain vertex SRVs.

Recommended: **(b)**. Terrain has its own vertex format, its own buffers, and one shared
hit group; keep the branch minimal.

## Pre-existing system to reconcile with: floating origin

`Scene` already has a floating-origin / camera-relative shift: `Scene::sceneOriginWorld`
(`XMFLOAT3` — **FP32**) and `prevSceneOriginWorld`, fed from
`Camera::getSceneOriginWorld()`. `RebuildTLASInstanceList()` subtracts it so the TLAS is
already built in a camera-local frame; `PrepareInstanceProperties` does the per-frame
refit equivalent; there is snap handling for motion-vector stability past ~500 m of
camera drift.

So the plan's "camera-relative TLAS" partly already exists. Reconciliation:
- Phase 1's FP64 `coordinate_system.h` is the **authoritative** world space.
- `sceneOriginWorld` becomes the FP32 down-cast of the planet/camera origin.
- Phase 5 terrain instance transforms = `(chunk.anchor_world - originWorld)` computed in
  FP64, cast to FP32 — using the **same** origin as `sceneOriginWorld`, so terrain and
  scene meshes share one consistent local frame.
- Planet FP64 origin = source of truth; `sceneOriginWorld` = its cast. Keep them in
  lockstep, including the snap logic.

## Decisions needed before Phase 4/5 (Phases 1–3 are unblocked)

1. Confirm deviation **A** — full-rebuild unified TLAS over max-sized buffers.
2. Confirm deviation **C** — terrain-specific surface-eval branch, option (b).
3. Queue creation — add `computeQueue` + `copyQueue` to `DeviceContext` (Phase 4
   prereq; the plan pre-authorizes this).

## Tagged sites (`// PLANET_INTEGRATION:`)

- `rdn/Core/DeviceContext.h` — struct that owns device/queue/fence/frames-in-flight
- `rdn/Core/DeviceContext.cpp` — command queue creation
- `rdn/Renderer.cpp` — per-frame TLAS update site; the clouds compute pass entry
- `rdn/Renderer_Pipeline.cpp` — `CreateTopLevelAS`; hit-group declaration
- `rdn/Scene/Scene.cpp` — `RebuildTLASInstanceList`
- `shaders/Pass_raygen_v8.hlsl` — HIT block (terrain material branch goes here)
- `shaders/Hit_v8.hlsl` — closest-hit stub (not the terrain hook)
