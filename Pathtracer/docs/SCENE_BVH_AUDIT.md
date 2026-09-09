# Loaded scene data and BVH audit

The original path was not optimal. It shared imported mesh BLASes and compacted
them, but duplicated loading data and rebuilt the active TLAS every frame. The
older `TopLevelASGenerator` refit methods were not used by the active renderer.

## Changes

- **Build inputs are allocated one mesh at a time.** The loader keeps CPU geometry;
  `CreateBlasBuildInputs` allocates vertex/index upload buffers just before that
  mesh's build. The build fence retires them immediately afterwards. Previously
  every imported mesh had a second full copy in upload heaps throughout loading.
- **Temporary copies end earlier.** Shared vertex vectors move into `MeshGPU`;
  merged single-use sources are released after baking. The source GLB byte vector
  is released after tinygltf3 has copied the data into its owned arena.
- **GPU copy lifetimes are explicit.** Light uploads survive until the flush that
  executes their copies. A compacted BLAS's source survives until the next build's
  fence, or the final flush. The old loop destroyed that source while its compact
  copy was still only recorded. At most one previous compaction source is retained.
- **Unchanged scenes reuse the active TLAS.** The builder compares complete
  descriptors in cached CPU memory: transforms, BLAS addresses, instance IDs,
  hit-group indices, flags, and instance count. It writes only changed upload
  descriptors and records no build when nothing changed. Movement still triggers
  a full build using the existing fast-trace policy. Terrain forces a rebuild,
  since streamed BLAS contents may change while an address is reused.
- **Capacity grows instead of dropping instances.** The scene-only TLAS grows
  geometrically and the renderer rebinds its SRV after a result-buffer replacement.
  Loaded terrain scenes reserve their scene-instance range before terrain IDs are
  fixed. Later attempts to overlap that fixed terrain range report an error.
  Out-of-range DXR IDs report errors instead of being truncated.
- **The redundant startup TLAS is no longer built.** Rays only use the unified
  orchestrator TLAS. Scene-only startup also avoids reserving unused terrain/rock
  capacity. Structural edits wait for prior GPU users before replacing instance
  resources and descriptors.
- **Procedural and material-variant mesh records use correct offsets.** Their
  material base now indexes the per-triangle material table, rather than using a
  material ID as a table offset. Procedural bounds are computed and shared-mesh
  bounds are copied for the scene bounds consumers.

## Verification

Run these commands from a Visual Studio Developer PowerShell:

```powershell
./tests/run_tlas_reuse_tests.ps1
./tests/run_gltf_instancing_tests.ps1
```

The TLAS test performs real GPU ray queries after reuse, movement, metadata edits,
removal, capacity growth and transition to an empty scene. It verifies that 100
unchanged frames record zero TLAS builds and preserve the result allocation.
The glTF tests also exercise deferred BLAS inputs and the production light-buffer
copies, including their staging lifetime. Renderer and orchestrator translation
units are compiled separately to cover the integration changes.

These checks establish correctness and removal of specific allocations/builds.
They do not establish a full-scene frame-rate improvement or a universally optimal
BVH layout.

## Remaining tradeoffs

1. **Moving scenes still rebuild.** A refit policy needs representative GPU timing
   of both build cost and ray traversal. Frequent refits can reduce build cost but
   degrade traversal as instances separate. Keep full builds for topology changes;
   benchmark refits and periodic rebuilds for transform-only changes. See the
   [DXR update constraints](https://microsoft.github.io/DirectX-Specs/d3d/Raytracing.html#acceleration-structure-update-constraints)
   and [NVIDIA's acceleration structure guidance](https://developer.nvidia.com/blog/best-practices-for-using-nvidia-rtx-ray-tracing-updated/).
2. **Small repeated meshes increase TLAS instance count.** Preserving them avoids
   geometry duplication; selectively merging spatially close copies can sometimes
   improve traversal. The best thresholds depend on the scene and GPU.
3. **CPU geometry is retained deliberately.** Global-buffer creation, emissive
   extraction, OMM rebaking and editing still consume it. Discarding it requires
   separating immutable geometry ownership from those consumers or adding reloads.
   Material-variant `CreateMeshInstance` also copies CPU/global geometry despite
   sharing the ray-tracing BLAS; a geometry/material ownership split is still useful.
   That currently unused helper also assumes opacity-compatible overrides: an
   override that changes opaque/alpha classification or an OMM alpha texture needs
   a rebuilt BLAS/OMM. Imported glTF instances retain their source materials.
4. **Light lookup storage scales with instantiated triangles.** Even non-emissive
   copies have full triangle-to-light ranges. Shared no-light ranges and cached
   per-mesh emissive lists could save substantial memory and rebuild work, but need
   coordinated handling of emission edits, instance offsets and fixed terrain ranges.
5. **BLAS build/compaction still synchronizes per mesh.** Batching small builds and
   compacted-size readbacks could reduce startup stalls; bound the batch's scratch
   and uncompacted memory so large assets do not recreate the peak-memory problem.

The next performance work should profile representative static, highly instanced,
moving-emitter and terrain scenes rather than infer optimal settings from mesh or
triangle counts alone.
