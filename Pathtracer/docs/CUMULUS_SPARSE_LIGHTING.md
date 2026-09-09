# Sparse cloud lighting and residual layers

September 8, 2026. Follow-up to the accepted cloud appearance, targeting residual thin-cloud layering, dawn flicker and lighting cost.

## Changes

The temporal density phase now spans a complete stratum instead of 80%. The old distribution never visited repeated gaps between strata. A common per-ray phase retains regular cell spacing. An experiment with independently offset nodes was rejected because it increased silhouette and guide error. The material, fine detail, density normals, sample counts and deterministic guide quadrature are unchanged.

Cloud lighting events now sample the inverse optical-depth CDF inside each linear-density cell instead of using its fixed median. Thick cells concentrate events near their visible front. This creates temporal source variation even when one opaque cell dominates the ray; it does not sample uniformly in physical distance or move the RR guides.

The 1–4 lighting budget now selects complete cloud sources: direct light and local shadows, all five scattering octaves, atmospheric hemisphere illumination and the ground term. A cheap three-anchor solar/sky estimate guides selection, but final lighting is evaluated at the selected location. Probability compensation preserves the mean of the discrete candidate integral. Budget 0 still evaluates cloud lighting in every occupied cell. Atmospheric source work inside the shell has a separate four-reservoir budget (two for cheap queries), sampling optical distance with RGB compensation. Air and cloud extinction are still integrated at every cell. The clear-air integrator outside the shell is unchanged.

The shadow cache previously stopped at optical depth 40 while returning values up to 80. Crossing that stopping threshold could abruptly add or remove a whole cell's extinction, which remains visible to the reduced-extinction multiple-scattering orders. Tracing now stops at the same 80-depth saturation used for storage. The source model and three local shadow strata otherwise remain intact.

## Verification

The 1920×1080 GPU suite passes, including secondary miss normalization, guide ownership, depth/motion, disabled and zero-coverage controls, inside/above clouds and floating-origin cases. All 18 production shader entry/variant compilations pass. Lighting convergence, atmosphere controls, optical integration and temporal-density/haze tests pass on the final source.

The new twilight regression scales extinction at fixed cache geometry and verifies linear optical depth until saturation. It fails on the preceding shader snapshot and passes after the stopping-threshold fix. The maximum measured half-float error is 5.96e-8 in the tested scaling case.

Against a 512-step reference, accumulated thin-cloud opacity RMS errors are:

| View | Previous | Current |
|---|---:|---:|
| Ground, upward | 0.02124 | 0.01829 |
| Cloud base | 0.01465 | 0.00924 |
| Inside layer | 0.06295 | 0.05638 |
| Dawn, sun facing | 0.03993 | 0.03610 |

All nine guide channels are bit-identical to the preceding version in these four comparisons. Lighting-budget changes also preserve all thirteen non-RGB channels at a fixed density/source phase. The material and normal functions were compared directly with the saved source and are unchanged.

A 65-position advancing-dawn sweep preserves extinction and guides at every position. This confirms separation from lighting, not that every perceptual flicker is gone. The sweep's normalized patch-brightness curvature was essentially unchanged (RMS 0.0630 before, 0.0638 after); it does not establish a general temporal-stability improvement. The cache discontinuity is independently reproduced and fixed, but distant sparse shadow sampling, the approximate scattering model and RR's response to advancing light can still matter. These tests do not run DLSS RR.

## Performance

Three alternating 32-frame batches at 1920×1080 on the RTX 5090, measuring primary colour plus stable guides, excluding RR:

| View | Previous | Current |
|---|---:|---:|
| Day | 6.53 ms | 6.25 ms |
| Sunset | 6.67 ms | 6.45 ms |
| After sunset | 5.88 ms | 5.90 ms |

This is a small saving, not the requested major reduction or a 3 ms result. Other paired runs were nearly equal, so do not promise these exact deltas in-engine. A pass-isolation measurement attributed approximately 3.6–3.8 ms to colour and 2.5–2.6 ms to stable guides. Full density reconstruction dominates many solid-cloud rays; reducing source evaluations alone does not remove that cost. No density, guide resolution or detail budget was reduced to manufacture a speedup.

## Artifacts and testing

Restart the existing executable to load the updated runtime headers. Start with the current settings and **Cloud lighting samples = 2**; 1 and 4 change source variance/work, and 0 is the all-cell cloud-lighting comparison. The atmosphere source estimator remains stochastic at 0.

`out/cumulus/sparse-final` contains final shader binaries, the 1080p smoke log, convergence/atmosphere/twilight/temporal/integration logs, dense-reference comparisons, moving-dawn data, paired timings and the runtime manifest. `out/cumulus/sparse-production` contains the production compilations. `out/cumulus/sparse-baseline` preserves the preceding source and test binaries. `sparse-v1` through `sparse-v5` are intermediate experiments, not the deployed source.
