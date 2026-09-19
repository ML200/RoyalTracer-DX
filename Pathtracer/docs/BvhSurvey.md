# BVH quality checks and Minecraft comparisons

The asset loader surveys models automatically during import. It keeps shared
meshes instanced, combines single-use submeshes that move with the same model,
then checks the combined geometry for spatial separation. Meshes with at least
4,096 triangles are split when a 16-bin surface-area estimate predicts at least
35% lower traversal cost, including an allowance for another BLAS traversal.
The existing per-BLAS triangle limit still applies. Triangle material IDs,
opaque/alpha ordering, reflected winding, and model transforms are preserved.

`[BVH survey]` log entries and `SceneModel::bvhSurvey` report the resulting
instance count, triangle count, and positive-volume root-bound overlap pairs.
Touching chunk faces are not overlaps. These are geometry diagnostics: DXR's
internal hierarchy is opaque, and predicted cost reduction is not a measured
frame-time improvement. Independently moving models remain separate.
Each unique mesh is scanned once; instance bounds use transformed root-box
corners, including the extra empty space introduced by rotation.

## Exact Minecraft survey

Build the opt-in `MinecraftBvhSurvey` target and compile its compute shader:

```powershell
cmake --build cmake-build-release-visual-studio --target MinecraftBvhSurvey
./include/dxc.exe -T cs_6_5 -E main -O3 tests/MinecraftBvhSurvey.hlsl -Fo out/probe.dxil
./cmake-build-release-visual-studio/MinecraftBvhSurvey.exe "WORLD" "PACK.zip" out/probe.dxil out/survey.json 128
```

The tool loads the whole world and the same pack and vanilla assets as the
renderer, meshes every selected node, and writes exact per-chunk CSV and JSON
totals. The first optional argument is the LOD factor; the camera defaults to
spawn plus 40 blocks. Three further arguments override camera XYZ.
It builds real DXR BLASes with `PREFER_FAST_TRACE` and measures deterministic
closest-hit and occlusion rays using GPU timestamps. It requires DXR 1.1 and
limits uncompacted BLAS storage to 6 GB; reduce the factor for larger scenes.

Traversal probes force opaque geometry and exclude textures, lighting and
shading. Their results depend on the camera being in a representative location:
a spawn inside a building or water can make rays terminate immediately. Check
hit counts and use full renderer captures before attributing an FPS difference.

## Full renderer capture

The following environment variables are opt-in, process-local comparison hooks:

| Variable | Meaning |
|---|---|
| `RT_MC_WORLD` | Override the configured Minecraft world's directory. |
| `RT_MC_CAMERA` | Six floats: eye XYZ followed by look-at XYZ. |
| `RT_BENCHMARK_FRAMES` | Render this many frames in a hidden window, then exit. |
| `RT_PERF_CSV` | Write CPU and per-pass GPU timings plus published scene counts. |
| `RT_LOG_FILE` | Existing console log destination. |
| `RT_SUN_HOUR` | Existing hook to hold the sun at a fixed hour. |

Launch comparisons sequentially with the executable's build directory as its
working directory. Use the same settings and camera protocol, allow streaming
to settle, and compare medians from the final frames. The GPU values describe
the previous completed frame; geometry counts describe the current frame. The
GPU frame interval covers the direct queue and excludes asynchronous queues
and presentation. It is not an FPS measurement.

Minecraft's light selection cache reuses an unchanged selection only when the
camera, rendered set, light configuration, available slots and light state are
unchanged. The Minecraft performance panel and CSV show whether it was reused.
