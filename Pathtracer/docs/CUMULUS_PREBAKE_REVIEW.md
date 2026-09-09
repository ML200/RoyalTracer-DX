# Additional cloud prebaking experiments

September 8, 2026. Follow-up to `CUMULUS_COST_REVIEW.md`. **No experimental shader is deployed.** The production sources and runtime remain the accepted conservative-density version, verified against the starting snapshot and runtime hashes.

## What is already baked

The Perlin-Worley body, Worley octaves, value noise and seam noise are generated once into two 256³ RG8 textures (64 MiB total). The per-sample material combines filtered texture values; it does not evaluate the original procedural noise generators. Atmospheric transmittance, atmospheric multiple scattering, cloud ambient hemispheres, broad cloud shadows and the reflection environment also already have caches or LUTs.

The remaining repeated work includes material coordinates/deformation, density composition, density marching, depth moments and six density evaluations per normal. A LUT can replace arithmetic yet add texture traffic, dependent reads and interpolation error. Timings below evaluate that tradeoff rather than assuming fewer formulas means a faster shader.

## Tested candidates

### Combined deformation texture

Three shifted value-noise reads were baked into one 256³ RGBA16F texture, adding 128 MiB. The original density material, detail scales and budgets were retained. This reduces the deformation lookup count but introduces a second interpolation over the shifted source fields.

Seven 960×540 views accumulated over 128 frames showed relative RGB RMS differences of 0.26–0.80%. More seriously, some normal guides changed substantially: the 99th-percentile angular difference was approximately 90–125° depending on the view. Density gradients in flat regions are sensitive to small field changes. This candidate was rejected before an extended performance study; it does not satisfy the guide-preservation constraint. The material-shape regression suite passing by itself was insufficient to establish equivalence.

### Depth-moment LUT

A 512² RG32F table (2 MiB) stored the first and second normalized moments of the existing eight-point optical quadrature. Its axes were logarithmic optical depth and a warped ratio of endpoint densities. Out-of-domain and very thin cells retained the direct calculation.

All seven views preserved RGB and cloud opacity exactly. Representative-depth RMS differences were about 2–7 mm; 99th-percentile normal changes were about 0.23–0.38°, with larger outliers in flat-gradient regions. The optical integration regression passed. The LUT was nevertheless slower in the paired primary measurement:

| View | Direct calculation | Moment LUT |
|---|---:|---:|
| Day | 5.133 ms | 5.197 ms |
| Sunset | 5.240 ms | 5.331 ms |
| After sunset | 4.778 ms | 4.835 ms |

Rejected: it adds interpolation and memory without a speed benefit.

### Multiple-scattering LUT and ambient layout

A 1024×128 R32F table (512 KiB) tabulated the existing five cloud scattering orders over optical depth 0–80 and view/sun cosine −1–1. The optical-depth axis used logarithmic spacing. The table was baked once with the noise; solar colour, disk visibility and direct self-shadowing remained live. The ambient table was also rearranged into contiguous hemisphere rows to replace two samples per hemisphere with one bilinear sample.

The 11,059,200-sample exact material regression passed. There was no reliable primary improvement:

| View | Existing lighting | Lighting LUTs |
|---|---:|---:|
| Day | 5.284 ms | 5.225 ms |
| Sunset | 5.290 ms | 5.281 ms |
| After sunset | 4.935 ms | 4.954 ms |

Synthetic secondary medians were also slightly worse: sharp 1.559 → 1.570 ms, roughness 0.5: 0.694 → 0.703 ms, diffuse 0.438 → 0.442 ms. These small mixed changes do not justify a new production resource/layout and an interpolation approximation. Rejected.

### Separate normal generation

A further scheduling experiment wrote guide depths first, then evaluated the existing six-point normal in a separate compute pass. It used the full density graph and full-precision scratch storage, with no new LUT. Combined timings again showed small mixed changes:

| View | Combined guides | Separate normals |
|---|---:|---:|
| Day | 5.381 ms | 5.371 ms |
| Sunset | 5.346 ms | 5.283 ms |
| After sunset | 4.955 ms | 4.995 ms |

The changed compilation context also changed some floating-point normal results. Rejected; the original guide pass remains intact.

## Method and limits

All paired timing tables use the verified RTX 5090 at 1920×1080, five alternating-order batches per view, four warmup dispatches and 32 timed dispatches per batch. Values are medians of batch averages. The primary measurement includes colour and complete guide generation. Each comparison runs the appropriate old/new cache, colour, guide and secondary-check shaders. Timings exclude DLSS RR, scene tracing and presentation; clocks are not locked. Compare the two columns within a study rather than absolute values across studies.

The timing evidence shows these particular prebakes are not a useful further optimization. It does not establish a general hardware bandwidth bottleneck or rule out other caching architectures. Baking the final compound density, rather than individual noise inputs, would require a spatial cache that handles moving weather, the spherical shell, detailed boundaries, footprint-dependent filtering and normals. It would need its own accuracy and invalidation work; no speedup or quality guarantee for that design has been established here.

## Artifacts

`out/cumulus/lut-baseline` preserves the starting shaders and affected test/application files. `lut-warp`, `lut-moments`, `lut-lighting` and `lut-normal-split` hold the experimental shader snapshots, binaries and logs. `out/cumulus/lut-study/summary.json` retains the individual timing batches; `restored-runtime-manifest.json` verifies the retained executable and runtime headers. The exact baseline shader, test and application source comparisons pass after restoring the experiments.

There is no new executable or shader update to test from this investigation.
