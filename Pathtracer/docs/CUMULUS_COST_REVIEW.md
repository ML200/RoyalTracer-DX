# Cloud density cost review

September 8, 2026. This revision optimizes the accepted sparse-lighting version without changing its density field, integration, lighting estimator or sample counts.

## Measurements

D3D12 timestamps on the default adapter, verified as **NVIDIA GeForce RTX 5090**, at 1920×1080. Five paired batches per view, alternating which version runs first. Each isolated pass has four warmup dispatches followed by 32 measured dispatches. Values below are medians of the five batch averages. Both versions run in the same process with their own colour, guide, shadow, environment and secondary shaders.

| Primary colour + guides | Approved version | Optimized | Reduction |
|---|---:|---:|---:|
| Daylight | 6.183 ms | 5.187 ms | 16.1% |
| Sunset, sun elevation −1.3° | 6.407 ms | 5.287 ms | 17.5% |
| After sunset, −3° | 5.912 ms | 4.811 ms | 18.6% |

Pass isolation identifies where the work goes:

| Daylight pass | Approved version | Optimized |
|---|---:|---:|
| Primary colour | 3.425 ms | 3.058 ms |
| Stable RR guides | 2.525 ms | 1.971 ms |
| Near/far shadow cache | 0.241 ms | 0.193 ms |
| Reflection environment cache | 0.216 ms | 0.181 ms |
| Atmospheric ambient bake | 0.099 ms | 0.100 ms |

At sunset, the shadow cache improves from 0.576 to 0.443 ms; the guide pass improves from 2.863 to 2.035 ms. Guide density reconstruction and its six density-gradient taps remain substantial even though the lighting source budget is only two. Reducing source samples alone cannot remove that work. The timing and the location of the successful optimization implicate repeated procedural material evaluation; these measurements do not distinguish memory bandwidth, texture latency, occupancy and arithmetic stalls.

The atmospheric ambient bake is reused by the engine until its settings key changes. It should not always be counted as recurring cloud-cache work. Isolated pass timings need not sum exactly to the combined measurement because their dispatch sequence and cache state differ.

The synthetic 1080p secondary-miss test also improves:

| Roughness | Approved version | Optimized |
|---|---:|---:|
| 0, sharp guide-producing reflection | 1.765 ms | 1.554 ms |
| 0.5 | 0.778 ms | 0.703 ms |
| 1 | 0.481 ms | 0.441 ms |

These secondary values are medians of five single dispatches with one queued miss per pixel. Actual engine cost depends on the number and distribution of misses. All timings are headless and exclude DLSS RR, scene tracing and presentation. GPU clocks were not locked. The 3 ms overall target remains unmet.

## Implemented optimization

Only `shaders/CumulusDensity_v8.hlsli` changes in production:

1. Read weather organization after the local cloud-top rejection. Samples above that top no longer fetch organization unnecessarily.
2. After the two body samples, compute a conservative upper bound on the remaining positive lobe and child-detail contributions. Return zero only when that bound is below the existing empty-field threshold, with a floating-point margin. Empty samples can skip up to six subsequent filtered noise lookups.

The broad lobe noise is UNORM, so `(noise − 0.43)` cannot exceed `0.57`. The wind-stretched octave similarly cannot exceed `0.70`. All their weights are nonnegative within the supported controls. Combining these maxima gives an upper bound on the pre-detail field. The remaining outward child contribution is below `0.260 * detailStrength`; the existing rejection uses `−0.27 * detailStrength`. The earlier test uses that same conservative envelope and an additional `1e−5` margin. Height metadata is computed before either early return.

Surviving samples execute the original material arithmetic. No wavelength, footprint fade, march spacing, interval budget, normal stencil, shadow sample count, cache resolution, lighting probability or temporal sequence changes. The accepted density can still reach its original maximum lookup count where needed.

Two other experiments were discarded: putting the six normal evaluations in a loop gave no reliable speedup and changed floating-point normals; skipping individual zero-weight texture contributions was faster but also changed some compiled results. Neither is in the deployed header.

## Quality and correctness checks

- A frozen pre-optimization material oracle matches **11,059,200 samples bit for bit**, including 1,667,522 nonempty samples. The 48 configurations cover different coverage, cloud scale, base/thickness, detail, distortion, seed and wind values; each probes fine/coarse paths and footprint fade boundaries. A self-comparison using the unmodified baseline also passes.
- Seven views at 960×540, accumulated over 128 frames, match **all 16 output channels bit for bit**: radiance, opacity, normal, depth, RGB transmittance, depth spread, reverse-Z, motion and guide opacity. Views include daylight, sunset, post-sunset, cloud base, inside the cloud layer, above it and a distant high view.
- The 1920×1080 GPU suite passes, including finite geometry, floating origin, cloud guides, wind motion, secondary radiance/reservoir normalization, roughness and guide ownership.
- Detail, twilight/cache saturation, atmosphere, optical integration, temporal sampling and lighting-convergence studies pass. All 18 production shader entry/variant compilations pass.

The material oracle uses separate candidate/reference shader entry points. Putting duplicate material calls in one kernel introduced different floating-point contractions even for an untouched baseline, so that arrangement was rejected as a bitwise oracle. The separate-entry baseline self-control validates the final test arrangement.

Exact comparisons cover the tested scenes and controls, not every possible camera and driver. The mathematical bound is what makes the skip conservative; the automated checks protect its implementation. This revision preserves the appearance and remaining limitations of the approved version rather than changing the lighting model.

## Reproduction and artifacts

From a Visual Studio Developer PowerShell:

```powershell
./tests/run_cumulus_tests.ps1 -DensityParity -OutputDirectory out/cumulus/density-parity
./tests/run_cumulus_tests.ps1 -Width 1920 -Height 1080 -OutputDirectory out/cumulus/gpu
```

`tests/CumulusDensityReference.hlsli` freezes the approved material for the parity test. `tests/CumulusDensityParity.hlsl` generates the probe points; the C++ runner checks exact results. The runner now prints its actual D3D12 adapter.

`out/cumulus/cost-baseline` preserves this revision's starting shaders. `out/cumulus/cost-final` contains final shader sources/binaries, validation logs, seven image pairs, timing data and the verified runtime manifest. `out/cumulus/cost-study/Profile.cpp` is the paired pass profiler; `Compare.cpp` produces the seven image pairs. `profile-summary.json` retains all five timing samples per result. `out/cumulus/cost-production` contains the production shader compilations.

Nsight GPU Trace was attempted but returned “No single-pass metric set selected” on this installation, including with an explicit Blackwell metric set. No hardware-counter or shader-stall conclusion is claimed. The cost breakdown above comes from D3D12 GPU timestamps and controlled pass isolation.

Restart the existing executable to load the updated density header. No application-code or resource-layout change is required.

The subsequent [prebaking investigation](CUMULUS_PREBAKE_REVIEW.md) tested deformation, depth-moment and lighting LUTs plus separate normal generation. None provided a reliable further saving while satisfying the quality constraint; this conservative-density implementation remains deployed.
