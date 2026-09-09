# Frostbite sky, atmosphere and cloud analysis

Analysis dated September 7, 2026. Based on the supplied text of Sébastien Hillaire's *Physically Based Sky, Atmosphere and Cloud Rendering in Frostbite*, the supplied GP Discord image, and the current working-tree implementation. Paper section numbers refer to that supplied version. This is a design review: no runtime changes or new GPU measurements accompany it.

Follow-up: [HanPi cloud article and source assessment](HANPI_CLOUD_ANALYSIS.md) examines a cheaper material organization and a diffuse multiple-scattering field as candidates for this architecture.

## Conclusion and measured constraint

Use the paper's separation of cloud shape, detail, lighting and smooth atmospheric transport. Keep our spherical geometry, actual secondary-ray origins and newer atmospheric model. The next iteration needs to reduce the work required to resolve a cloud, rather than simply add noise octaves or decrease the existing step slider.

The latest user measurement is **about 1.1 ms Cloud cache and nearly 11 ms Cloud rays at 1920 × 1080, DLAA**. That is approximately 12.1 ms across these timers, superseding the earlier 7–8 ms estimate. Rays account for roughly 91%. Reaching 2 ms requires about a sixfold reduction overall; cache optimization alone cannot do it.

Timer scope matters: Cloud rays includes the primary cloud/clear-atmosphere pass and selected PT secondary misses. Final composition and RR are outside it. The separate transmittance/multiple-scattering LUT bake is also outside these two cloud timers. Measure cloud-disabled atmosphere at the same view to establish the incremental cloud cost, but preserve the full reported timing as the user-visible baseline. See [timer assignment](../rdn/Renderer.cpp) and [UI description](../rdn/Editor/Editor.cpp).

The paper's Xbox One figures are not directly comparable: §5.10 uses **16 samples, two multiple-scattering orders, half-resolution main clouds and quarter-resolution planar reflections**, with clouds covering three fifths of the screen. Those figures support reducing spatial and integration work; they do not establish the cost of our full-resolution arbitrary PT misses and RR guides.

## Where our work goes

These are source-level operation counts, not a GPU profile of individual bottlenecks. Early exits, visible coverage, cache behavior and compiler optimization affect actual cost.

| Current behavior | Implication |
|---|---|
| [Full material](../shaders/CumulusDensity_v8.hlsli) uses up to 13 filtered 3D noise lookups: organization, top height, three warp components, two base samples, uplift lobes, wisps, two seam samples and two fine samples. | Every march, local shadow and guide can repeat a substantial material evaluation. Even fully dense cores can fetch fine noise whose final contribution is zero. |
| [Primary integration](../shaders/CumulusRender_v8.hlsli) defaults to 64 radiance steps, then a separate 48-step deterministic guide march; the normal adds six density evaluations. | Up to 118 full material evaluations for a single cloud interval before local shadow work. The guide march has 75% as many sampling sites as the main march, **not necessarily 75% of its GPU time**. |
| The main march uses exponential spacing and opacity termination, without a conservative occupancy hierarchy. | Long empty intervals still consume steps. Thin occupied intervals can be missed despite the large total budget. |
| Outside the local light volume, each occupied shading point falls back to 10 sun-density samples plus four upward samples, followed by a detailed local shadow sample. | Distant and horizon rays can turn one view march into nested procedural marches. The fallback has to be measured separately. |
| [Secondary misses](../shaders/Pass_cumulus_secondary_v8.hlsl) default to 48 steps and use the camera-pixel angular footprint regardless of path roughness. | Broad diffuse lighting can pay to resolve details that its angular integral will average away. Sharp first reflections can also request the full guide march. |
| Atmospheric medium, sun transmittance and atmospheric multiple scattering are evaluated inside every cloud step, including empty samples. | Smooth air transport is repeatedly evaluated at the cloud detail rate. |
| [Light bake](../shaders/Pass_cumulus_light_v8.hlsl) visits 128 × 32 × 128 voxels every frame. [Environment bake](../shaders/Pass_cumulus_environment_v8.hlsl) integrates 512 × 256 directions at up to 24 steps. | Cache work is substantial, but the measured 1.1 ms makes it the secondary priority. |

The 256³ RGBA8 noise texture occupies 64 MiB. Merely increasing texture resolution is unlikely to solve either the modeling or runtime problem. Conversely, blindly shrinking it risks losing the detail we are trying to recover.

## Techniques to adopt or adapt

### 1. Cheaper shape evaluation, with explicit cloud structure

**Paper §§5.3–5.4 and Appendix D.** Separate a weather field, a type-versus-height profile, precombined low-frequency shape noise and precombined erosion noise. The paper explicitly reports a bandwidth benefit from storing its fixed noise combinations as single-component textures.

For our material, precombine fixed octave sums instead of repeatedly rebuilding them from the RGBA field. Move organization and top-height variation into a shared low-frequency weather representation. Keep controls that actually change morphology; do not bake away all independent scales just to minimize fetch count. A few scalar samples plus selectively evaluated detail is an implementation target to validate, not a promise of identical appearance.

The reference image suggests four coordinated features:

1. A fairly flat condensation base with distinct, irregularly spaced rising towers.
2. Large asymmetric domes with nested rounded lobes, rather than uniformly rough capped masses.
3. Shaded creases between lobes, with increasingly fine detail near visible boundaries.
4. Thin wisps and broken edges whose scale and direction differ from the thick cores.

Our broad height taper begins reducing the profile above 30% of the local layer height. Changing the noise frequencies alone will not produce the reference's hierarchy of individually developed towers. Introduce spatially varying development/type profiles: shallow cumulus, tall congestus and transitional forms. Place medium lobes within the large updraft structure; use small detail to modify their boundaries and extinction. Lighting must see enough of that same medium structure for the folds to remain visible inside the silhouette.

A later weather map can supply coverage, condensation altitude, thickness/development, type weights and wind. Use continuous planet-space fields with local high-resolution detail; a single globally tiled horizontal map is insufficient for the requested spherical weather model. This global mapping is our extension, not something the paper fully solves.

### 2. Spend samples where there is cloud

**Paper §§5.4–5.6 motivate cheap coarse shape and low sample counts; the following occupancy design extends it.** Add conservative coarse occupancy or density bounds, then traverse empty cells cheaply and refine occupied boundaries. Share that representation with the guide pass and shadow traversal.

The bound must include domain warp and the outward wisps added before thresholding. Skipping whenever the unmodified base is empty would erase those wisps. An average-density mip is not a safe empty-space bound. Filter fine noise by projected footprint, retaining the current distinction between pixel footprint and march-step length.

Keep fresh stratified radiance and local-shadow samples for RR. Lower sample counts should produce recoverable sampling variation, but jittering heterogeneous density inside an exponential does not automatically give an unbiased transport estimator. RR cannot recover systematically erased features, wrong mean transmittance or unlit folds. Validate reduced budgets against a converged reference rather than treating visible noise as proof of correctness.

### 3. Reduce guide cost without making guides follow radiance noise

**RR-specific adaptation; not covered by this paper.** The separate 48-step full-detail guide march is a major candidate for improvement. Use occupancy-assisted deterministic traversal, locally refine the extinction region that determines the representative depth, and derive a compatible gradient from the same representation. Investigate precomputed or analytic gradients instead of six additional complete material calls.

A cheaper macro-only depth may visibly disagree with wisps. Reusing stochastic radiance moments may destabilize motion and disocclusion. Neither should become the default without comparisons of depth, opacity ownership, wind motion and reflected motion against the current deterministic implementation. Preserve the full field where local refinement shows it matters.

A participating medium has no unique surface normal or surface depth. Our extinction-weighted position, density gradient, albedo and roughness are **volume proxies for a surface-oriented interface**, not an exact volumetric G-buffer. Maintain coherent position/depth/motion and preserve the reflector's guides when a reflected cloud is being denoised.

NVIDIA requires consistent input-resolution guide resources; its RR guide also documents no dynamic-resolution support. If cloud shading uses a smaller grid, reconstruct dense inputs at the fixed RR extent first. Treat the optional color-before-transparency input as an experiment, not documented volumetric support. Keep RR as the temporal stage. [NVIDIA RR integration guide](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS_RR.md)

### 4. Bound lighting cost across near clouds and the horizon

**Paper §§5.5.2 and 5.8.** Combine coarse long-range optical depth with a small stochastic near-field shadow correction. That is compatible with our existing direction, but the out-of-cache procedural fallback defeats its bounded cost.

Use world-stable multiscale shadow coverage, such as nested local volumes plus a coarser distant representation. Keep enough near-field resolution to shadow the reference's lobes. Bound far-field work even when the sun grazes hundreds of kilometres of cloud shell. Do not copy the paper's four shadow samples without accounting for those planetary paths.

For higher-order scattering, compare two or three of the paper's scattering octaves against the current dimensional-profile approximation, sharing optical-depth work across orders. The paper admits that its approximation lacks true spatial light spreading. Constraints on each octave's albedo are useful, but do not make their summed result a complete energy-conserving multiple-scattering solution. A small converged cloud PT reference should judge thick-cloud brightness, edge illumination and shadowed folds.

Keep our liquid-droplet HG–Draine phase as the initial single-scattering baseline. The paper's two-lobe HG discussion (§5.7) is useful for understanding backlit and sun-behind-camera appearances, not a reason to automatically replace the current fitted phase. Test lighting from behind, from the side and toward the sun at several optical depths.

### 5. Move smooth atmosphere work out of detailed cloud sampling

**Paper §§3.5, 5.9.1–5.9.2.** Cache aerial perspective and evaluate it at representative cloud depths or a small number of cloud segments. This directly addresses our repeated air evaluation inside the detailed march.

A single representative depth is a useful fast path when a compact cloud lies behind clear foreground air. It is approximate for thick clouds, overlapping layers, horizon views and a camera inside the volume. Keep a coarse mixed-medium integration or segment-based fallback for those cases. Compose foreground air, cloud scattering/transmittance and background once; avoid double-applying atmospheric attenuation to an already integrated sky.

A camera-frustum aerial-perspective volume cannot correctly answer arbitrary off-frustum rays from reflection origins. Secondary paths need an origin-aware parameterization or a cheap finite-segment integration.

The current atmosphere includes Earth shadow but does not attenuate its direct in-scattering by cloud sun visibility inside the cloud march. Its atmospheric multiple-scattering field also remains a clear-sky field. The paper's cloud/atmosphere coupling offers a useful improvement: directional cloud shadow for direct air lighting, and a low-frequency cloud-modified illumination field for diffuse atmospheric scattering. This should improve overcast darkness and distant blue-shadowed clouds. Avoid applying one scalar cloud factor to all lighting directions or feeding cloud radiance back twice.

Keep the existing triangular ozone layer and newer transmittance/multiple-scattering LUT approach. The paper's older ozone distribution and reduced LUT parameterization have explicit compromises. Hillaire's later reference implementation provides a more suitable atmosphere comparison and a volumetric path-tracing reference. [Author's 2020 implementation](https://github.com/sebh/UnrealEngineSkyAtmosphere)

### 6. Make secondary quality depend on the transported angular footprint

**Adaptation of the paper's separate main/reflection budgets, §§5.5.3 and 5.10.** Retain actual ray origins and directions for sharp reflections. Add path-cone or equivalent footprint information so rough and diffuse misses can use fewer samples and appropriately filtered density/lighting.

Nearby clouds still require parallax and local visibility. Use coarse origin-aware transport or local probes there; use a filtered distant environment where the approximation is appropriate. Substituting the old low-resolution angular cache for every secondary ray would reintroduce the blockiness and incorrect reflected placement the user already rejected.

Lower-resolution primary cloud shading remains an option after these changes, but raw screen-space downsampling of unrelated reflection rays is not a valid reflection reconstruction. Spatial reuse must respect origin, direction, depth and lobe. Evaluate quality at DLAA as requested, without hiding costs or artifacts behind a lower global render resolution.

## Low-risk cache cleanup

These changes help the frame budget, but cannot solve the measured 11 ms ray cost alone.

| Finding | Proposed change |
|---|---|
| [Atmosphere LUT bake](../rdn/Renderer_Pipeline.cpp) runs every frame. Its tables cover altitude and sun zenith angle, rather than only the current camera and sun. | Rebuild when atmosphere coefficients, radii or other actual bake dependencies change. Camera motion and ordinary time-of-day changes do not inherently require rebaking these tables. This saving lies outside the cloud timers. |
| [Cloud ambient table](../shaders/Pass_cumulus_ambient_v8.hlsl) covers all sun angles and a range of cloud altitudes, yet runs every frame. | Rebuild for atmosphere, cloud base/thickness and baked intensity changes. Moving the current sun alone does not change its contents. |
| The environment bake still writes a second layer of depth moments. The current shader search finds only `CumulusEnvironment(rayDir, 0u).rgb` as its consumer. | Remove the unused depth layer and environment-only moment calculations after checking any debug consumers. The saved second layer is 1 MiB; this is cleanup, not a large claimed speedup. |
| The separate clear-sky view bake remains in the PT pass sequence when clouds are enabled. | Audit consumers, then split the required sun-state update from potentially redundant clear-sky texture work. |
| The light volume is camera-relative and recomputed every frame. | World-stable placement enables incremental updates later. Do not simply skip frames while changing the coordinate basis used to read old voxels. |

## What to retain and what not to copy literally

- **Retain analytical homogeneous-step integration.** Paper §5.6, equation 17, is already implemented as the source integral proportional to `(1 - exp(-extinction * distance)) / extinction`. It improves integration within a homogeneous step, not the spatial resolution of the underlying density.
- **Retain extinction-weighted depth moments.** Do not replace them with an unqualified average of sample positions or assume a single mean depth exactly describes multilayer transparency.
- **Retain atmosphere-colored ambient light.** The paper's zeroth-order ambient plus height gradient is explicitly artistic. Its lower cost is attractive; its lack of direction and occlusion would regress our twilight behavior if copied literally.
- **Retain finite primary intersections and actual reflected origins.** The paper lists important geometric simplifications and remaining limitations; it is not a complete arbitrary-ray planetary transport design.
- **Do not stack its temporal EMA in front of RR.** Adopt economical sampling and coherent motion, and let RR handle temporal reconstruction.
- **Check photometric consistency before changing constants.** Paper §4's sunlight calibration and disk solid-angle normalization are useful for coherent sun/sky exposure. Our atmosphere and direct sun use separate intensity scales. Calibrate them together; copying lux or per-metre extinction values into kilometre-based normalized cloud density would not by itself fix dusk.

## Proposed implementation order and acceptance

1. **Measure the 11 ms more finely:** separate primary, secondary and guide work; count occupied samples and out-of-cache lighting fallbacks. Establish a fixed 1080p DLAA scene and matching clouds-disabled baseline. No new profiling was run during this analysis while the user was testing.
2. **Restructure density and traversal:** precombine fixed noise, add conservative occupancy, and use it for cheaper stable guides. Preserve the reference's large and medium structure before tuning microdetail.
3. **Bound illumination and specialize secondary work:** multiscale shadow coverage and path-footprint-aware sampling. Remove repeated smooth atmosphere work with an appropriate mixed-medium fallback.
4. **Apply dependency-based cache updates and dead-output cleanup.** Keep a separate total-frame account of atmosphere savings outside the cloud timers.
5. **Tune type profiles and multiple scattering against references.** Use identical fields, exposure, phase and camera for approximation-versus-PT comparisons.

Acceptance should cover front/side/back lighting, sun below and above the horizon, ground/inside/above-cloud views, moving camera and wind, thin wisps over geometry, sharp and rough reflections, diffuse misses, and origin rebases. Compare raw mean radiance/transmittance as well as the final RR image. Stable guides alone do not establish convincing denoised clouds.

The supplied image is a reasonable morphology and lighting target. It does not reveal the renderer, sample count, GPU or timing, so it cannot establish feasibility at 1–2 ms. The measured gap calls for architectural reductions in per-ray work. No particular speedup or match to path-traced multiple scattering is established by this analysis.
