# Cloud performance investigation

Updated September 8, 2026. RTX 5090, 1920x1080 input. Preserve the approved material and RR guides while improving lighting and cost.

## Latest: atmosphere interaction

The atmosphere revision preserves density and guides, evaluates air scattering at existing cloud density steps, and uses the existing near/far cache for atmospheric direct-light shadows. Cloud higher orders now use broad optical depth, and ambient fill samples upper/lower atmospheric hemispheres. The diffuse atmosphere table concentrates its existing resolution around twilight and low altitude. See [research and limits](CLOUD_ATMOSPHERE_LIGHTING.md).

Isolated paired 1080p means are 5.18 -> 5.16 ms in daylight, 5.38 -> 5.42 ms at sunset and 4.52 -> 4.27 ms after sunset. Cache totals are 0.73 -> 0.79, 0.91 -> 0.97 and 0.25 -> 0.32 ms respectively; they include the additional ambient hemisphere bake, which the engine reuses until invalidated. This is not a 3 ms solution. The 17 production compilations, 1080p GPU suite, lighting convergence and atmosphere/twilight regressions pass. Deterministic guide values match the prior revision exactly; tiny transmittance/accumulated-opacity rounding differences are documented in `CUMULUS_SCENE.md`.

Use `out/cumulus/atmosphere-final/perf-isolated.log`; the earlier `perf.log` overlapped another GPU test. The standalone comparison cannot establish RR visual quality. Daylight raw noise decreases, but sunset changes are mixed. The following sections retain measurements from earlier revisions.

## September 8 twilight and cache revision

The accepted material and gradient functions are unchanged. Cloud optical depth no longer contains a discontinuous planet blackout; solar visibility is evaluated separately. Broad distant sun shadows use fixed quadrature, while local shadows and view integration retain stochastic samples. Opacity-stratified lighting reservoirs preserve the discrete source integral in expectation. Ambient fill retains light in dense clouds and uses the upper hemisphere's band areas.

The main performance improvement shares one density evaluation per cache voxel across its vertical column, replacing four upward evaluations per voxel. The near/far grids integrate 32/16 height cells, respectively. Sun work is omitted for fully shadowed cache regions only with a conservative interpolation/solar-disk margin, and primary samples omit local sun work when their solar disk has fully set.

Final paired means (three alternating 32-frame runs, each version's own caches, renderer closed):

| Condition | Primary before -> after | Cache before -> after |
|---|---:|---:|
| Daylight | 4.99 -> 4.83 ms | 1.13 -> 0.73 ms |
| Sunset, -1.3 degrees | 5.25 -> 5.05 ms | 1.05 -> 0.91 ms |
| After sunset, -3 degrees | 4.95 -> 4.26 ms | 0.37 -> 0.25 ms |

The 3 ms total target remains unmet. These are headless shader timings, not RR frame timings; the harness also rebakes ambient light where the engine normally reuses it. Full 1080p GPU checks, all 17 production shader targets, ten-condition source convergence, exact guide invariance and new solar-disk/vertical-cache regressions pass. The new planet-shadow regression fails on the approved pre-fix shader as expected. Shapes and all 13 non-RGB outputs match the approved baseline in seven comparison views.

Normalized raw temporal RMS is lower in tested sunset views, but improvements vary (about 37% for the ground view and 9% for the nearby side view); the distant daylight subset is about 9% noisier. These measurements include varying density/shadow samples and changed mean lighting, and do not establish final RR stability. The ambient and multiple-scattering models remain approximations. See `CUMULUS_SCENE.md` for validation details and current limitations.

Artifacts: `out/cumulus/sunset-final/{perf,noise,visual}.log`, `sunset-final-tests.log`, `sunset-final-lighting.log`, `sunset-final-twilight.log`, `sunset-production`, `sunset-old-regression`. The paired runner is `out/cumulus/sunset-study/SunsetStudy.cpp`. Loop/indexing alternatives under `sunset-loop`, `sunset-loops`, `sunset-mass`, `sunset-index` and `sunset-unroll` did not establish a better final primary cost; they are not production alternatives. The approved snapshot and historical reference remain untouched.

## Approved appearance baseline: medium-sized detail

The user accepted the medium-detail version on September 8: "Keep this. It looks superb". Preserve this appearance in subsequent performance work. The density header SHA-256 is `a0a65824ea78c526b858e40a86363b8ada6ac01fd465d3604d78986bb58e77b9`; matching source/runtime copies and a local shader snapshot are recorded in `out/cumulus/approved-medium-billows-a0a65824`.

The two erosion lookups now supply 3.2-frequency child cells and 8.4-frequency breakup, about 2.27x / 2.82x larger than the rejected tiny features. The detail displaces a signed density boundary and shares the accepted body wind deformation, local shadows and guide evaluation. No extra noise lookup or resource is added, but the expanded empty-boundary envelope and dependent lookup can change cost.

The full 1080p GPU suite, material/wind/silhouette checks, lighting convergence with exact guide invariance across budgets, and 17 production shader checks pass. The user was running the renderer during validation; current timings are contended. Do not reuse the preceding isolated timings below as a measurement of this version. No new performance claim or visual retuning was made after acceptance. See [CUMULUS_SCENE.md](CUMULUS_SCENE.md) for current details and artifacts.

## September 8 silhouette correction

The wind deformation previously ran after the two main body samples, faded with fine erosion and was omitted from coarse shadows. It now runs before body density sampling and remains active for distant rays and shadow caches. Existing billow wavelengths are preserved. Primary lookup count is unchanged; coarse shadow density adds the three broad warp lookups required to follow the form.

Controlled three-pair 1080p comparison with each version's own primary/cache/environment shaders: **5.36 -> 5.35 ms primary**, **0.86 -> 1.14 ms cache**. The primary cost is essentially unchanged; the cache costs about 0.28 ms more. No 3 ms total claim is made. Logs: `out/cumulus/silhouette-benchmark.log`; source: `out/cumulus/detail-before/SilhouettePerf.cpp`.

Material and projected-opacity tests now require the body, distant representation and shadow cache to move with distortion. The previous shader fails the new broad-deformation regression for the expected reason. The new shader, full 1080p GPU suite, sparse-lighting convergence with exact budget-independent guides, 17 production shader checks and application build pass. See [CUMULUS_SCENE.md](CUMULUS_SCENE.md) for the current behavior and limitations.

## September 8 follow-up: restore detail scale

The smaller cellular lobes and extra micro-density/fringe terms were removed in response to visual feedback. The recovered body and lobe frequencies now receive coherent wind shear and bending from existing coarse warp values. Edge coordinates stretch downwind; they do not increase transverse frequency. This restores the earlier detail scale without extra density lookups. The original 12 m minimum gradient footprint returns.

Three local shadow strata remain, but use a dynamic loop. Unlike the prior density graph, this graph regressed badly with unrolling (ground preview about 6.8 ms at 1280x720). Changing only the loop removed that spike; a one-shadow-tap variant was also measured but not retained. The reason for the compiler-sensitive cost is not established by hardware counters. Variant logs are under `out/cumulus/detail-wind-variants`.

Final controlled three-pair 1080p comparison: previous smaller-detail shader **5.83 ms**, wind-deformed broad detail **5.20 ms**, about **11% faster**. Near-view cache ~0.85 ms, sharp/rough/diffuse secondary ~1.16 / 0.72 / 0.48 ms. Full GPU checks, lighting convergence with exact budget-independent guides, bounded material and wind-reversal checks pass. The application and 17 production shader checks pass. The total 3 ms target is still unmet. Current implementation, limitations and artifacts are in [CUMULUS_SCENE.md](CUMULUS_SCENE.md).

## Earlier September 8 detail revision

Small billows now use a signed density boundary, cellular lobes distorted by existing micro-noise, patchy wind-stretched breakup and softer optical fringes. The added texture lookup is restricted to the translucent boundary. Three stratified local shadow samples replace one at nonzero fine detail; the near normal footprint becomes 6 m. Sparse source sampling and cheap secondary lighting remain unchanged.

A controlled three-pair 1080p comparison measures the September 7 default at **4.12 ms**, versus **5.80 ms** for the final turbulent revision. This adds **1.68 ms**, reflecting a quality tradeoff. The total 3 ms target remains unmet. Material tests show added outward structure with about -0.53% integrated-density change and exact distant/coarse density under the fine-detail toggle. The final lighting convergence study still passes, including exact guide invariance across lighting budgets. Full details and the historical reference precision differences are in [CUMULUS_SCENE.md](CUMULUS_SCENE.md).

Two local attempts were not retained: changing the three-tap shadow loop from unrolled to dynamic did not improve timing; hoisting the boundary mask and tightening the early-out envelope caused a large primary-pass regression in the rounded prototype. The final implementation uses the simple conservative expanded boundary. Compiler scheduling/register pressure are possible explanations, not established hardware-counter findings. Local logs are `out/cumulus/detail-loop-benchmark.log` and `detail-rejected-bound-tests.log`.

## September 7 revision: sparse source sampling

The RR-assisted sampling candidate below is now implemented. Keep the 64-sample density march and 48-sample deterministic guides; use 1-4 weighted reservoirs to evaluate the existing air/cloud lighting source after each shell crossing. The default is two. The proposal mass is the maximum RGB component of the view integral weight times scattering. Inverse-probability weighting preserves the discrete source sum in expectation. Density and shadow-candidate random sequences are independent of selection randomness.

The complete source still includes the restored local/procedural sun shadowing, atmospheric colour, profile-dependent multiple scattering, ambient bands and ground contribution. No replacement cached-lighting formula is used for primary clouds. Atmospheric lighting *inside* the cloud layer joins the estimator; external clear-air intervals retain their existing integrator. Secondary rays retain their cheaper cached integration without another reservoir.

A small additional density octave uses the previously unread body channel of the existing micro-noise RG fetch. It requires no additional lookup or texture memory. Its independent control defaults to one, fades with footprint and preserves empty/saturated regions. Setting both new controls to zero restores the prior density and complete-lighting formula (RGB roundoff differs, while all tested opacity/transmittance/guide bits match).

Controlled current comparison at 1920x1080: previous shader **5.84 ms**, current full lighting **5.24 ms**, one/two/four selected samples **3.91 / 4.17 / 4.78 ms**, two samples with fine detail **4.19 ms**. Three alternating 32-frame measurements per mode put the default about **28% below the previous shader**. The older measurements below were taken under different observed GPU conditions; do not compare their 8-9 ms values directly with this run.

The GPU convergence study covers seven poses/light conditions, including sun-facing backlighting. Default two-sample RMS error after 512 independent selections is 0.74-1.25%; worst mean channel error is 0.012%. All 13 non-RGB channels remain bit-identical to complete lighting at the same detail setting. One/four-sample controls also converge in ground, dusk and backlit tests. This proves neither a converged volume path tracer nor RR image quality: it validates the new estimator against the existing discrete model. The full 1080p GPU suite, 17 production shader checks and application build pass.

Current details, commands, caveats and artifacts are in [CUMULUS_SCENE.md](CUMULUS_SCENE.md). The 3 ms total budget is still unmet. Density and deterministic guides remain the main unthinned work.

## Earlier investigation

## Implemented: two noise planes

The RGBA8 256-cubed texture is now two RG8 textures: body/Worley FBM and value/seam. A density lookup uses one component; it now fetches two-component texels instead of four. Procedural functions, dimensions, wind, sample counts and total noise storage (64 MiB) remain unchanged.

The extra texture has explicit SRV/UAV slots, initialization and transitions. Bindless textures follow these descriptors; the secondary query remains at slot 79. Both planes are written in one bake dispatch. Primary radiance and all guides remain **bit-identical in all 14 reference cases**. The frozen reference and strict comparison test were not relaxed or refreshed.

A controlled comparison alternates old and new primary kernels in the same GPU runner, with identical noise and cache values:

| Trial | Packed RGBA8 | Two RG8 planes |
|---|---:|---:|
| 1 | 8.827 ms | 8.560 ms |
| 2 | 8.894 ms | 8.882 ms |
| 3 | 9.472 ms | 8.440 ms |
| Mean | 9.064 ms | 8.628 ms |

Each timing averages 32 dispatches. All six images match exactly. The roughly 5% gain is smaller than some differences between independent runs; use this controlled result. Separate full-harness cache rebuilds fell from roughly 1.1-1.2 ms to 0.84-0.87 ms. These are standalone shader timings, not complete RR frames.

## Cost breakdown

Initial ablations disable individual work. Their images are deliberately incomplete. Timing differences include scheduling/register effects and are not additive costs.

| Ground-view diagnostic | Time |
|---|---:|
| Complete primary | 9.19 ms |
| Without guide march and normals | 5.96 ms |
| Without normals | 7.73 ms |
| Without shadow evaluation | 6.86 ms |
| Guides only | 1.92 ms |

One detailed density evaluation can perform 13 filtered lookups: organization/top, three-axis warp, two body scales, uplift, wisps, two seam scales and two fine-detail scales. The radiance march, local shadow taps, 48-sample guide march and six-point normals repeat this work. Outside the near light cache, procedural sun and upward shadow marches add more evaluations.

## Rejected experiments

- Four R8 planes were faster, with byte-identical stored noise. Small sampling-roundoff differences occasionally amplified into changed normals on nearly flat regions, especially at orbital distances. Two RG8 planes retained exact output instead.
- Separate radiance and guide dispatches matched their reference but cost about 9.45 ms, versus about 7.06 ms for the corresponding combined scalar-texture test. The guide-only ablation did not predict an end-to-end saving. Lost locality and scheduling overhead are hypotheses, not established hardware-counter findings.
- A 16x4 thread group gave only a small isolated change within timing variation. 32x2 and 4x16 were slower. Production remains 8x8.
- Keeping a separate RGBA texture only for guides preserved guide values but added 64 MiB and lost the primary gain.
- Earlier primary distant-cache experiments changed illumination without a reliable speedup. The original primary shadow fallback remains.

## Next substantial candidates

1. **Conservative empty-interval skipping.** Bound the restored density graph, including domain warp, positive uplift/wisp terms and spherical weather coordinates. Skip only intervals proven empty. The former simplified-material bounds are invalid for this field. Validate maximum detail, coverage, wind, altitude and finite ray endpoints before enabling it. Potentially removes work from both radiance and guides; no speedup estimate is established yet.

2. **Cheaper density gradients.** Prototype derivatives of the existing trilinear noise and density functions to replace six complete material evaluations per normal. Compare against current normals, including flat regions and orbital precision. This changes the gradient estimator and needs numerical and RR inspection; it is not a proven saving yet.

3. **Reduce repeated shadow work.** Instrument local-cache hits, detailed local taps and procedural fallbacks. Test adaptive cache refinement or conservative shadow bounds while preserving the penetration model and twilight illumination. Simply replacing all fallback rays with a coarse cache already failed the appearance/performance comparison.

4. **RR-assisted radiance sampling: implemented above.** Density and guides remain at their restored budgets. Lighting is estimated with fewer weighted samples, with convergence checks for thin wisps, twilight and backlighting. Actual RR reconstruction still needs manual inspection.

The 3 ms total target likely requires fewer complete material evaluations, beyond repacking their inputs. No unimplemented candidate has a measured guarantee of meeting that budget.

## Artifacts and verification

The production/reference GPU suites and strict comparison commands are in `CUMULUS_SCENE.md`. Final output: `out/cumulus/paired-final1080`. Frozen reference: `tests/cumulus_reference`. The application build, 17 production shader entry/variant checks and GPU checks pass.

Local ablation sources, runner and log remain in `out/cumulus/profile`. The controlled comparison is recorded in `out/cumulus/paired-final1080/alternating.log`; its source is `out/cumulus/profile/PairedBenchmark.cpp`. That runner uses retained old shaders and noise readback, so it is a local investigation artifact rather than a portable replacement for the checked-in test suite.


## September 8 video follow-up: shared extinction and coherent sunset lighting

The current implementation and measurements are in [CUMULUS_SAMPLING_REVIEW.md](CUMULUS_SAMPLING_REVIEW.md). The separate 48-step guide march is removed; deterministic linear density nodes now supply cloud opacity and eight-point depth moments. Smooth diffuse lighting is evaluated through the ray, sparse samples handle detailed direct shadows, and near/far shadow caches blend continuously. The accepted material and normal functions are unchanged. Final paired primary timings improve about 20-27%; the 3 ms total budget remains unmet. These results supersede the earlier current-performance and guide-budget descriptions above.


## Temporal-noise follow-up

The user's subsequent screenshot exposed frozen density grain and atmospheric shadow slices. [CUMULUS_TEMPORAL_REVIEW.md](CUMULUS_TEMPORAL_REVIEW.md) documents temporally rotating colour/source quadrature, a separate stable RR guide pass, measured convergence and its added cost. Those measurements supersede the current-runtime claims above.
