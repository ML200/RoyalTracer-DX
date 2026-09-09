# Cumulus scene: approved density and current transport

Updated September 8, 2026. The user-approved medium child billows and whole-body wind distortion remain the density baseline. Conservative empty-sample rejection now avoids unnecessary material lookups while preserving the evaluated field and normal stencil. The original appearance snapshot is `out/cumulus/approved-medium-billows-a0a65824`, with density-header SHA-256 `a0a65824ea78c526b858e40a86363b8ada6ac01fd465d3604d78986bb58e77b9`. It is an artifact backup, not a Git commit or a recording of live UI settings.

The current build enables an experimental nearby cache of the complete density field. See [the density-cache experiment](CUMULUS_DENSITY_CACHE.md) for timings, interpolation error, wind fallback and the comparison toggle. The procedural material described below remains the reference.

## Current sampling and illumination

The renderer reconstructs linear extinction between density nodes whose sampling phase changes each frame. It integrates each cell's cloud opacity analytically and samples cloud lighting from its optical-depth CDF, concentrated near the front of optically thick cells. This replaces the former randomly placed, constant-density cells, whose deep samples could illuminate an entire thick segment at sunset.

Primary rays use the configured 64 intervals near the camera, smoothly increasing to 96 for distant shell entries. The near spacing scale stays 2 km instead of growing with distance. Fractional interval counts avoid grid jumps at integer budget changes. A spatial hash plus temporal low-discrepancy phase covers the full width of each stratum and lets RR accumulate density/edge error. The former 80% phase span left repeated gaps in the temporal sample distribution. The approved density field itself is unchanged.

A separate compute pass preserves the approved deterministic linear reconstruction for guide opacity, depth and depth spread. Split eight-point quadrature resolves within-cell moments. Colour sampling cannot move those guides. Sharp secondary guides use at least 16 intervals; secondary colour retains 4-12 cheap intervals and cached lighting. Both primary passes are included in the Cloud rays timer.

The sparse cloud budget now covers the complete source: direct shadows, all cloud scattering octaves and atmospheric sky fill. Each opacity band selects a source using throughput and a cheap three-anchor solar/sky importance estimate, with probability compensation. The anchors only guide selection; the selected source uses the full lighting model. Within-cell events sample the extinction CDF rather than a fixed median, producing temporal lighting variation even in a solid cloud. Atmospheric source evaluation within the cloud shell uses a separate four-reservoir budget (two for cheap queries) and RGB-compensated optical sampling. Extinction and RR guides remain independent of lighting selection. These estimators approximate transport through a piecewise reconstructed density field, not full heterogeneous path tracing.

Sun transmittance and directional cloud shadows use the centroid of the visible solar disk near the planet horizon. The original disk fraction still controls energy. Cloud shadow integration uses fixed distance strata and clips both shell intervals without redistributing samples when the ray starts intersecting the planet. Near and far shadow caches blend continuously; the separate ten-sample distant procedural fallback has been removed. Three stochastic local shadow strata retain detail over the first 240 m. Cached sun optical depth now stops at its stored cap of 80. Previously stopping at 40 but returning values up to 80 caused discontinuous multiple-scattering light as the sun moved.

The atmospheric upper/lower hemispheres and broad optical-depth scattering octaves still supply diffuse light. There is no fixed blue tint or nighttime light floor. Air and clouds share the same accumulated camera transmittance. Clear-air source positions now rotate within fixed extinction cells, using an optical-depth proposal with RGB compensation. This removes stationary atmospheric shadow slices while keeping clear-air transmittance independent of source jitter. LUT bakes retain their deterministic sampling. Within each cell, air uses average cloud extinction; cloud source lighting samples the cell optical distribution. Higher orders, spatial cloud-to-cloud transport and RGB twilight colour remain approximations.

## Try it

Restart `cmake-build-relwithdebinfo-visual-studio/Pathtracer.exe`. This update rebuilds the executable and deploys the density-cache shaders together; runtime files are verified in `out/cumulus/density-cache-study/runtime-manifest.json`. It adds a density bake pass, cache resources and two camera-buffer fields while retaining the 72-byte cloud settings block.

Open **F1 > View > Sun / Time of Day > Cumulus**, then **Daylight cumulus scene** if a preset is needed. Existing settings can be retained for the sunset comparison.

- **Cache cloud density (experimental):** on by default. Reuses nearby full-detail density, with a 27-frame population sweep. Turn off to compare the procedural field. Nonzero wind uses procedural density automatically.
- **Cloud lighting samples:** 2 by default; 1 for fewer complete cloud lighting evaluations, 4 for less raw variance, 0 to evaluate cloud lighting at every occupied interval. The atmospheric source budget, density and guides are independent of this choice.
- **Sky samples:** 64 by default, with the smooth distant increase described above.
- **Reflection samples:** 12 by default, decreasing with roughness; sharp guided queries have a minimum of 16.
- **Billow distortion:** 1 by default, range 0-2; controls the entire wind-deformed body.
- **Lobe and edge detail:** 1 by default, preserving the accepted medium detail.
- **Wind X / Z:** set deformation and material-advection direction. Reversing wind reverses the lean; this is not a convection simulation.

Other defaults: base 1.5 km, thickness 3.6 km, coverage 0.28, scale 0.8, extinction 14/km, seed 17, stationary wind. Settings changes reset RR and SHaRC through the existing change path. The two RG8 noise planes, existing caches and 72-byte settings block are retained.

## Approved density

The recovered weather organization, height profile and broad body wavelengths remain. Wind shear and broad bending act before both main body samples, then uplift, seams and detail use the same deformed coordinate field. This preserves the approved silhouette behavior in primary rays, distant views, rough reflections and shadow caches.

The new detail replaces the two previous erosion lookups with a paired breakup lookup and a cellular lookup. Child-cell frequency is 3.2, versus 7.27 in the rejected tiny-cell version: the base cells are about 2.27 times larger. Breakup frequency is 8.4, versus the former 23.7 grain: about 2.82 times larger. The paired breakup channels bend the child cells before density evaluation, and the existing wind coordinates stretch them downwind. This avoids adding another tiny-frequency rim.

The density boundary stays signed until medium detail has displaced it, so child lobes can extend outward rather than only erode an already clipped shape. For UNORM inputs, the remaining positive contribution is below `0.260 * detailStrength`; early rejection uses a conservative `0.27 * detailStrength` envelope. Saturated interiors keep their early exit. Broader seam erosion remains, followed by a small mean-centered breakup term gated by `4*d*(1-d)`.

Child detail uses the existing 12-90 m material-space footprint fade. Smaller breakup fades over 8-65 m. These fades apply to medium detail, not the whole-body wind deformation. Coarse shadow density omits the medium layer but retains the deformed body; detailed local shadow samples and RR gradients evaluate the full field. The accepted material and its maximum noise lookup count are unchanged.

The subsequent cost optimization moves the organization lookup past the local-top rejection and bounds the maximum positive contribution of the remaining lobes before sampling them. It skips only provably empty material work. Surviving material arithmetic and the normal stencil retain their original form. Eleven million material probes and seven before/after rendered views match bit for bit; see the cost review for the scope of those checks.

## RR guides

The guides preserve the deterministic extinction reconstruction from the preceding version, evaluated in their own compute pass. They remain independent of temporal colour sampling, lighting samples and time-of-day changes. The six-point density gradient uses the existing 12-70 m footprint.

| Guide | Representation |
|---|---|
| Depth | Reverse-Z at the extinction-weighted position, using the shared 0.01 m to 100,000 km range. |
| Normal | Normalized outward density gradient of the deformed field, with radial fallback. |
| Diffuse albedo | Linear single-scattering albedo, RGB 0.999. |
| Specular albedo / roughness | Zero / one for the primary cloud proxy. |
| Primary motion | Camera and wind motion of the representative position. |
| Reflected motion | Finite virtual hit for a selected first sharp reflection, including wind and the previous reflector transform. Primary reflector material/normal guides remain its own. |
| Depth spread | Extinction-weighted deviation, retained for diagnosis. |

A volume has no unique surface depth or normal; these remain coherent proxies for RR's surface inputs. Opacity determines ownership against geometry (default threshold 0.5) and sky (0.015). Broad secondary misses omit moment and normal evaluation. Albedo, roughness, depth, normal and motion are not derived from a selected noisy lighting point.

## Verification and limits

The 1920x1080 D3D12 suite, lighting convergence study, twilight/cache integration tests, cloud/atmosphere controls, material/wind/silhouette checks, and all 18 production shader entry/variant compilations pass. The integration probe independently checks optical CDF inversion, depth moments and visible-sun geometry. A 65-position advancing-sun sweep preserves all non-RGB channels. A separate 512-interval reference comparison reports lower distant opacity, depth and silhouette errors in all three tested viewpoints.

See [the density-cache experiment](CUMULUS_DENSITY_CACHE.md) for current validation and the additional measured 5–10% primary-ray saving. [The cost review](CUMULUS_COST_REVIEW.md) records the preceding exact optimization and its measured 16–19% primary-ray saving. [The sparse-lighting follow-up](CUMULUS_SPARSE_LIGHTING.md) and [the temporal sampling follow-up](CUMULUS_TEMPORAL_REVIEW.md) record preceding revisions. [The preceding sampling measurements](CUMULUS_SAMPLING_REVIEW.md) document the static-noise version. These are headless shader tests without DLSS RR; the in-engine visual result still needs manual review. Surface-style RR guides are volume proxies, not exact volumetric denoiser features. Dense distant clouds can still show integration error, and coarse far shadows cannot reproduce full volumetric path tracing.

From a Visual Studio Developer PowerShell:

```powershell
./tests/run_cumulus_tests.ps1 -Width 1920 -Height 1080 -OutputDirectory out/cumulus/gpu
./tests/run_cumulus_tests.ps1 -LightingStudy -OutputDirectory out/cumulus/lighting
./tests/run_cumulus_tests.ps1 -TwilightStudy -OutputDirectory out/cumulus/twilight
./tests/run_cumulus_tests.ps1 -AtmosphereStudy -OutputDirectory out/cumulus/atmosphere
./tests/run_cumulus_tests.ps1 -IntegrationStudy -OutputDirectory out/cumulus/integration
./tests/run_cumulus_tests.ps1 -DetailStudy -OutputDirectory out/cumulus/detail
./tests/run_cumulus_tests.ps1 -TemporalStudy -OutputDirectory out/cumulus/temporal
./tests/run_cumulus_tests.ps1 -DensityParity -OutputDirectory out/cumulus/density-parity
```

Historical studies remain in `CUMULUS_OPTIMIZATION.md` and the existing `out/cumulus` artifacts. The nine frozen sources in `tests/cumulus_reference/` were not changed. Global weather maps, other cloud families, evolving convection and attenuation on finite secondary segments remain future work.
