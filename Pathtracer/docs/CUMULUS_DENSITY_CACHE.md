# Combined cloud density cache experiment

September 8, 2026. Implements the spatial cache proposed after the [prebake review](CUMULUS_PREBAKE_REVIEW.md). Enabled by default for manual comparison. The procedural material remains the reference; cached density is an approximation, not a bit-identical optimization.

## Try it

Restart `cmake-build-relwithdebinfo-visual-studio/Pathtracer.exe`. In **F1 > View > Sun / Time of Day > Cumulus**, toggle **Cache cloud density (experimental)**. Allow about 27 frames to populate a stationary window after a material change. Toggling resets RR and SHaRC history. Use zero Wind X/Z to exercise this first implementation; animated wind automatically uses procedural density.

Compare cloud edges and close fly-throughs, including movement during cache population. Lighting retains the existing stochastic sampling, atmosphere integration and shadow model. No static lighting noise is introduced.

## Cached field and fallbacks

The complete unit-density material is baked, including organization, deformation, lobes, erosion and medium detail. Eligible samples replace that compound evaluation with a tag check and trilinear lookup. Existing noise textures generate each vertex.

- Each world-aligned brick owns eight cells and nine vertices per axis, including duplicated borders.
- A 48 x 24 x 48 brick window maps to a 432 x 216 x 432 R16F atlas. RGBA32I tags contain signed world-brick coordinates and a material epoch.
- Voxel spacing is `0.024 km * max(cloudScale, 0.2)`: 19.2 m at default scale. The default window spans about 7.37 x 3.69 x 7.37 km around the cloud layer above the observer.
- Texture payload is 77.73 MiB, excluding allocation alignment. Resources remain allocated with the checkbox off.
- A frame checks at most 2,048 bricks. Valid bricks skip computation; missing bricks evaluate the original material. A sweep takes 27 phases. Barriers publish density before consumers can use its tag.

Only fine samples on the material's full-detail plateau use the cache: footprint at most `0.008 km * max(cloudScale, 0.2)`. Filtered/coarse samples, absent or stale blocks, and nonzero wind retain procedural evaluation. Epochs change with coverage, base, thickness, scale, detail, wind, seed and distortion. Lighting, extinction and time of day remain live. Shader reload clears tags through the noise bake.

Signed arithmetic handles negative world coordinates. The bake uses `precise` density evaluation to prevent compiler reassociation discrepancies between independently compiled vertex evaluations at planet-scale coordinates. R16F storage still quantizes density.

The six-point normal stencil uses the procedural gradient. Interpolated extinction can slightly change guide depth and thus the position where that gradient is evaluated. Albedo, motion, ownership and reflection-guide conventions remain intact. Guides remain deterministic after the relevant bricks populate.

## Measurements

RTX 5090, 1920 x 1080. Primary values are medians of five alternating-order batches, each with four warmup and 32 timed dispatches. Both colour and complete guide generation are timed against the prior production shaders. Population, other caches, scene tracing, presentation and DLSS RR are excluded. Clocks were not locked.

| View | Procedural baseline | Cached density | Reduction |
|---|---:|---:|---:|
| Day | 5.202 ms | 4.659 ms | 10.4% |
| Sunset | 5.180 ms | 4.945 ms | 4.5% |
| After sunset | 4.877 ms | 4.521 ms | 7.3% |

Synthetic secondary tests, one miss per pixel, showed median sharp-query time of 1.815 -> 1.070 ms, roughness 0.5: 0.696 -> 0.635 ms, and diffuse: 0.438 -> 0.424 ms. These use five single-dispatch samples per setting, not the primary batch procedure, and do not predict a particular scene's reflection workload.

Cold population summed to approximately 0.98 ms over 27 phases in the default probe; an already populated sweep summed to about 0.11 ms. The app spreads this work across frames. Cost varies with density, material settings and movement. The total 3 ms cloud budget is not reached.

## Accuracy and verification

Seven 960 x 540 comparisons accumulated 128 samples with matched cameras, lighting and sample sequences. Relative linear-RGB RMS differences: daylight 0.276%, sunset 0.092%, after sunset 0.038%, near cloud base 1.465%, inside 1.634%, above 0.032%, tested high view zero. Image-wide errors can hide larger local edge differences. The cache is not certified visually lossless; close views need manual RR comparison.

The D3D12 regression covers cold/partial population, world identity despite slot wrapping, weather invalidation, negative-coordinate movement, reuse, filtered footprints, wind and the off switch. All 230,400 probe points hit after full population. Stored vertices match precise procedural evaluation within 0.000489 absolute density, consistent with R16F store conversion. Interpolated unit-density samples have roughly 3.1-4.7% relative RMS error in the tested windows.

The full-HD cached suite passes finite radiance, transmittance/opacity bounds, unit normals, reverse-Z projection, motion, reflection guides, stochastic radiance with stable guides, dawn/dusk, finite segments, wind fallback, floating-origin rebasing, inside/above-cloud views and disabled/zero coverage. The original material parity test passes 11,059,200 bit-identical samples; that validates the unchanged procedural fallback, not cached interpolation.

The application builds. All 19 production shader variants compile. The actual production root-signature construction is separately serialized and checked for descriptor offsets; six cache/secondary pipelines are created with it. New SRVs use t57/t58 and UAVs use u45/u46. Existing u34/u35 resources retain their bindings.

In a VS Developer PowerShell:

```powershell
./tests/run_cumulus_tests.ps1 -DensityCacheStudy -OutputDirectory out/cumulus/density-cache-final
./tests/run_cumulus_tests.ps1 -Cached -Width 1920 -Height 1080 -OutputDirectory out/cumulus/density-cache-final
```

## Artifacts and limits

`out/cumulus/density-cache-baseline` preserves the previous shaders and relevant application/test sources. `density-cache-final` contains test binaries and comparison renders. `density-cache-study` contains timing batches, summaries, image comparisons, build logs, production-root validation and `runtime-manifest.json` for the rebuilt executable and deployed shaders.

This first cache does not accelerate advected wind or cover an entire planet simultaneously. Coarse/distant filtering remains procedural. Interpolation can soften nearby detail and may become visible when switching between cached and procedural samples. The checkbox retains the original field for comparison. Visual feedback will determine whether to refine resolution, footprint transitions, movement updates and animated-weather support.
