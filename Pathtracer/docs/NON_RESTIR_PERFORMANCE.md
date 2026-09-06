# Non-ReSTIR performance work — 2026-09-06

Implemented shader and pipeline optimizations without lowering resolution,
samples, bounce limits, training density, cache confidence or guiding quality
settings. The requested 20–30% **whole-frame FPS improvement remains unverified**.
The numbers below are component measurements, not frame-rate predictions.

## Changes

- PT camera rays no longer clear/finalize unused reservoirs or append survivors
  to the ReSTIR-only queue. This removes 80 bytes of reservoir stores per pixel,
  plus the queue atomic and four-byte store per surviving pixel. At 1920×1080,
  the reservoir clear alone was about 158 MiB per frame.
- The host skips the queue-header clear in PT mode and skips the immediately
  trailing barriers of inactive passes. Barriers between active producers and
  consumers remain. Returning to ReSTIR suppresses temporal reuse for its first
  frame so it cannot read reservoir history left over from an earlier run.
- PT primary-extra reads can use the GPU cache. This path does not require the
  globally coherent scratch/reload behavior of ReSTIR.
- Training evaluates the full BSDF and its diffuse share in one traversal, for
  both direct lighting and continuation. GGX/coat omit transmission calculations
  when no enabled lower lobe uses them. Rendering computes layer transmission
  only after a successful cache lookup.
- SHaRC marks deposited slots in a 128 KiB bit mask. Resolve loads one 32-byte
  mask per group and touches accumulation/history only for marked slots. It
  retains one thread per slot, preserving throughput when every slot updates.
  The extra atomic is included in the accumulate-plus-resolve measurements.
- Cache and guide reset/eviction invalidate compact state words; allocation
  already initializes every payload field before publishing a new key.
- The bilateral reflectance test uses one logarithm instead of three, with the
  same positive-reflectance distance formula.

The dirty mask is appended after the existing cache and guide tables. Their
addresses and capacities remain unchanged. **Restart the rebuilt application**:
hot-reloading these shaders into an older executable does not enlarge its buffer.

## Measured GPU costs

RTX 5090, driver 616.64. D3D12 timestamp queries; eight repetitions with one warm-up,
reporting the best of the remaining repetitions. Baseline and optimized runs
were sequential. Other desktop applications were open, so these are local
microbenchmarks rather than an isolated full-scene benchmark.

The maintenance fixture uses 1,048,576 occupied cache slots and varies the
fraction receiving one observation. It deliberately tests both sparse updates
and the fully updated worst case.

| Work | Before (ms) | After (ms) |
|---|---:|---:|
| Resolve, no updates | 0.10355 | 0.00307 |
| Resolve, 1/128 slots updated | 0.10883 | 0.00944 |
| Accumulate + resolve, 1/128 updated | 0.11754 | 0.01853 |
| Resolve, 1/16 slots updated | 0.12534 | 0.02995 |
| Accumulate + resolve, 1/16 updated | 0.13488 | 0.03930 |
| Resolve, every slot updated | 0.48675 | 0.48842 |
| Accumulate + resolve, every slot updated | 0.63194 | 0.63235 |
| Reset cache and guide tables | 0.50925 | 0.00304 |
| 1M cold scattered lookups | 0.02582 | 0.02582 |
| 1M warm surface lookups | 0.03293 | 0.03290 |

Sparse accumulation plus resolve takes 71–84% less GPU time in these fixtures;
the all-updated case is essentially unchanged. Reset is infrequent, so its
savings must not be counted as a steady-state frame-time improvement.

The material fixture uses analytic LUTs and eight coherent material classes.
1M separate full-plus-diffuse evaluations took 0.01350 ms before; the fused
version took 0.01206 ms after. Ordinary full evaluation was 0.01203/0.01232 ms,
and evaluation with the diffuse lobe removed was 0.01216/0.01194 ms. These small
arithmetic measurements exclude texture access, tracing and SER live-state costs;
they do not establish an overall material-pass speedup.

## Verification

- RelWithDebInfo application build succeeded.
- Production PT, training, prepare, resolve, camera, ReSTIR raygen/shift,
  hit/miss/any-hit, shading, primary clouds and postprocess shaders compiled.
- All SHaRC GPU regressions passed: accumulation, thin and parallel surfaces,
  collision rejection, reset/eviction, sparse history, corrupt-history recovery,
  reconstruction, long-path roulette, floating-origin rebases and guiding.
- Added sparse-mask tests across word/group boundaries and the end of the table,
  untouched-history checks, and a second resolve that must not replay deposits.
- Added 16,384 material/reflectance cases, including coated/sheen, anisotropic,
  metallic, solid/thin glass, grazing angles and removed diffuse lobes. Fused
  values/PDFs are compared with separate evaluation; reflectance support is
  compared with the previous three-logarithm expression.

Run from a Visual Studio Developer PowerShell:

```powershell
./tests/run_sharc_tests.ps1 -OutputDirectory ./out/sharc-tests
```

The same fixtures can benchmark an earlier shader snapshot:

```powershell
./tests/run_sharc_tests.ps1 -ShaderDirectory ./out/non-restir-perf/baseline/shaders -OutputDirectory ./out/sharc-baseline
```

Local source snapshots, compiled test binaries and logs are under
`out/non-restir-perf`. The snapshot includes the working settings/cache capacity
present before this work; it is not a clean Git HEAD comparison.

## Remaining scene validation

Windows computer-use access to Pathtracer was not approved, so scene appearance
and a controlled before/after frame-rate comparison could not be verified.
Use the same camera, scene, internal resolution, samples, bounce settings, cache
warm-up and DLSS settings. Disable frame generation and frame-rate limits for
the comparison. Record cold and warm results separately, including the existing
prepare/train/resolve/PT GPU timers; inspect both raw and denoised output.
The camera and pipeline savings above are implemented but not timed in-scene.
