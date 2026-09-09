# Editor and rendering pipeline

The default remains **Path tracer + SHARC**, with the existing sampling and
diffuse-resampling defaults.

## Editor

- **View** contains Scene, Camera, Materials, Environment, Integrator and DLSS.
- **Integrator** includes samples per pixel, path limits, SHARC and diffuse
  resampling. Cache tuning, guiding, inspection and GPU timings start collapsed.
  Selecting legacy ReSTIR reveals its applicable controls.
- **Diagnostics** contains the executed pass list, DLSS buffer inspector and
  performance graphs. The pass list shows the previous frame's recorded work;
  inactive passes can be included explicitly. UAV barriers are omitted from the UI.
- **Experimental** contains the optional DLSS 5 integration.
- The former Initial Sampling window and disabled NRC controls were removed.
  The DLSS preset selector exposes Default/D/E/F from the bundled SDK; the
  report-only jitter override and unsupported preset letters are no longer exposed.
- Selecting a DLSS buffer displays it immediately. Off or closing the inspector
  restores the prior output view. SHARC inspection takes priority if both are set.
- CPU fence waiting is labelled **GPU wait**, rather than GPU frame time.
  Cache/path/cloud timings continue to use actual GPU timestamp queries.

`IntegratorSettings` is the shared C++ settings type. Existing shader constant
slots and render defaults are unchanged. ImGui's original window IDs remain in
the renamed titles so existing window positions and sizes survive.

## Pipeline changes

Pass feature requirements are classified when `PassSystem::Build` parses the
pipeline. Each frame tests integer masks instead of hashing the legacy shader
names. A skipped producer still skips its trailing barrier; required barriers
remain in order. DLSS guide barriers are submitted together in one API call.

Scenes with no emissive triangles skip the PT mesh-light prefetch pass and its
barrier. Both render and SHARC training paths skip mesh-light NEE, while sun and
environment lighting remain enabled. A shared host/HLSL flag uses previously
retired bit 7, without increasing root-signature size. The light sampler returns
an invalid sample without accessing absent buffers; malformed trees and invalid
point samples also terminate safely. Populated zero-power trees keep their
existing uniform fallback distribution.

Transport edits, including integrator/cache changes, material overrides, texture
filtering and path limits, reset incompatible DLSS history. The diffuse material
override also resets SHARC. Cloud inspection and guide-threshold edits reset
reconstruction as needed while retaining the learned lighting; changes to cloud
shape or lighting still invalidate the cache.

These changes remove specific dispatches, buffer accesses and CPU work. They do
not establish a full-scene frame-rate gain. Compare equal settings and GPU
timestamp measurements when profiling; CPU fence wait is not a substitute.

## Verification

From a Visual Studio Developer PowerShell:

```powershell
./tests/run_render_pipeline_tests.ps1
./tests/run_light_tree_tests.ps1
./tests/run_sharc_tests.ps1
./tests/run_gltf_instancing_tests.ps1
```

The pipeline tests cover PT/SHARC, uncached PT, legacy ReSTIR, diffuse reuse,
cloud bake gates, absent mesh lights and history invalidation. Light-tree tests
cover empty sampling with null resources alongside real GPU sampling/PDF checks
through the full trail depth. The other suites exercise the production cache,
guiding, diffuse resampling, instancing and surface/light shaders.

For manual UI review, check Integrator, Environment and DLSS at the saved window
sizes, switch integrators, open and close buffer inspection, and compare the
recorded pass list with the selected features.
