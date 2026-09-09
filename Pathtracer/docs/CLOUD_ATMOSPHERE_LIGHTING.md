# Cloud–atmosphere lighting research and implementation

September 8, 2026. Preserve the approved cumulus density and RR guides. This revision changes transport and illumination, including the distant horizon case reported by the user.

## Research conclusions

For reference quality, use coupled volumetric path tracing of air and droplets: shared camera extinction, cloud shadows on atmospheric sunlight, and indirect paths between clouds and the surrounding air. High-albedo, strongly forward-scattering clouds are difficult to converge. Disney's *Deep Scattering* predicts higher-order radiance from spatial density descriptors, illustrating why local density alone cannot represent that illumination. This is a quality reference, not evidence for our 3 ms budget. [Kallweit et al., 2017](https://arxiv.org/abs/1709.05418).

The practical production approach is a mixed model: ray-marched aerial perspective and direct shadows, with cached or approximated higher orders. Epic explicitly supports cloud shadows in the atmosphere and per-sample atmospheric light transmission, and uses scattering octaves for clouds. Its cinematic aerial-perspective options also acknowledge the quality limits of coarse reconstructions. [Epic volumetric cloud documentation](https://dev.epicgames.com/documentation/en-us/unreal-engine/volumetric-cloud-component-in-unreal-engine).

Hillaire's scalable atmosphere approach and its path-tracing comparison implementation remain useful for the clear-air component. We keep the existing transmittance and isotropic multiple-scattering tables, but change the latter's coordinates to concentrate resolution at low altitudes and twilight. That remapping is our implementation choice, not a claim that the paper prescribes it. [Author implementation](https://github.com/sebh/UnrealEngineSkyAtmosphere), [paper](https://sebh.github.io/publications/egsr2020.pdf).

The Oz approximation sums scattering orders with progressively reduced shadow extinction, contribution and phase anisotropy. Its authors describe the improvement obtained by also sampling an indirect path. We use five higher-order terms with cached broad optical depth, keeping detailed local shadows in the direct term. This removes the old local-profile and view-penetration gates. It remains a tunable approximation and does not conserve transport energy exactly. [Wrenninge, Kulla and Lundqvist, 2013](https://fpsunflower.github.io/ckulla/data/oz_volumes.pdf).

More recent directions include spatiotemporal reuse of volumetric paths and learned radiance caches. Volumetric ReSTIR resamples multidimensional paths; our current per-ray source reservoirs do not implement it. GSCache (2025) trains a Gaussian radiance cache for scientific volume visualization. Neither cited result establishes 3 ms planetary procedural clouds at 1080p on this engine. [Volumetric ReSTIR](https://graphics.cs.utah.edu/research/projects/volumetric-restir/), [GSCache](https://arxiv.org/abs/2507.19718).

Spectral atmosphere rendering is a separate, worthwhile improvement for dusk colour. García's four-wavelength approximation explicitly emphasizes sunrise/sunset during fitting. Adopting it requires consistent spectral extinction, sun transmission, scattering and conversion back to the engine's colour space; changing only the cloud tint would be inconsistent. This revision retains the existing RGB atmosphere. [García, updated January 2026](https://fgarlin.com/blog/spectral-sky/).

## What the investigation found

- Combined air/cloud extinction and ordered compositing were already present. A 1,024-step numerical reference found approximately 0.45–2.12% aggregate radiance RMS difference from the 12-step clear-air march over the tested distance/elevation sweeps, with maximum transmittance difference about 0.0055. This does not validate every horizon configuration, but did not support globally increasing clear-air samples.
- Air inside the cloud shell competed with droplets for two source reservoirs. Even an empty shell therefore had lighting-selection noise. It was unbiased relative to the discrete model; RR still had to reconstruct that noise using cloud-oriented guides.
- The 32x32 diffuse atmosphere table used linear solar cosine and altitude, leaving almost four degrees between the nearest sunset samples and about two kilometres between altitude samples.
- Diffuse cloud fill used only the upper sky, and a local density gate imprinted each small billow into twilight illumination.
- Atmospheric direct scattering ignored cloud shadows.

## Implemented

Air scattering is evaluated at every existing cloud density step with the same accumulated RGB throughput and combined extinction. Only expensive cloud illumination uses the opacity-stratified reservoirs. Clear-air segments also receive cloud shadows, using spherical projection into the existing near/far optical-depth cache. The global atmospheric multiple-scattering term remains available in shadow.

Ambient illumination now includes upper and lower atmospheric hemispheres. A broad, high-albedo transport approximation uses upward and downward column depths; it has no local density gate. Full column depth is retained beyond the direct-light cutoff so subtraction cannot falsely expose thick interiors to the lower hemisphere. No new texture, descriptor, history buffer or settings field is needed.

The atmosphere table now uses signed-square-root solar cosine and square-root altitude coordinates. Bake sampling concentrates near the ray's lowest altitude. Bake and consumers share the mapping. Both caches invalidate through existing startup/settings paths; restart the renderer after updating shaders.

## Verification and limits

The 1080p GPU suite, 17 production shader targets, ten-condition lighting convergence study, solar-disk/column tests and new atmosphere regressions pass. Tests cover cloud-shadowed air, invariant camera extinction, empty-cloud controls, haze independence from cloud lighting budget, broad diffuse response, LUT coordinate round trips and columns exceeding optical depth 80. The density header is unchanged from the preceding revision. All nine deterministic guide channels match that revision exactly in seven comparison views; transmittance differs only by floating-point rounding (maximum about 1.1e-7), and accumulated cloud opacity by at most 1.7e-5.

Raw 32-frame normalized RMS improves in daylight and the tested side sunset view, but rises about 6% in the sun-facing ground sunset view. This is not a universal stability fix. Headless comparisons cannot validate DLSS RR output or reproduce the user's exact camera/settings from a screenshot.

The model still lacks spatial transport between separate clouds, feedback from cloud-scattered radiance into atmospheric multiple scattering, and spectral twilight transport. The coarse far cache limits shadow detail; the two-hemisphere ambient closure can flatten or darken broad cloud masses. Those limits require visual feedback. See `CUMULUS_SCENE.md` for measured cost and reproducible checks.

Artifacts: `out/cumulus/atmosphere-baseline`, `atmosphere-final`, `atmosphere-production`, and the bounded comparison/reference tools in `atmosphere-study`. The original approved morphology snapshot and frozen historical reference were not changed.


## September 8 moving-sun follow-up

The video follow-up adds visible solar-disk centroid directions, fixed-distance shell shadow quadrature, continuous near/far broad shadows, shared linear cloud extinction and RR moments, and separate deterministic diffuse / sparse direct illumination. See [the implementation measurements](CUMULUS_SAMPLING_REVIEW.md). The atmosphere still uses RGB tables and the cloud diffuse model remains an approximation; these sampling fixes do not establish equivalence to coupled volumetric path tracing.


## Temporal-noise follow-up

The user's subsequent screenshot exposed frozen density grain and atmospheric shadow slices. [CUMULUS_TEMPORAL_REVIEW.md](CUMULUS_TEMPORAL_REVIEW.md) documents temporally rotating colour/source quadrature, a separate stable RR guide pass, measured convergence and its added cost. Those measurements supersede the current-runtime claims above.
