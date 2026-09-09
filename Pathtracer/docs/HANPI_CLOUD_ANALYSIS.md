# HanPi cloud article: assessment for RoyalTracer

September 7, 2026. Follow-up to [the Frostbite analysis](FROSTBITE_CLOUD_ANALYSIS.md). Analysis only; no renderer changes, imported implementation or new GPU measurements.

## What the article contributes

The linked article is **“体积云各向同性多重散射近似” — “An approximation of isotropic multiple scattering for volumetric clouds”**, by hulalalala. Its `hp_phi_fwd` term approximates the broad diffuse illumination remaining after many scattering events. It combines slow attenuation, a source-establishment weight and boundary/underside heuristics. It supplements directional scattering rather than replacing the whole lighting model. The author explicitly describes a one-dimensional approximation, not a full three-dimensional diffusion solution. [Article](https://zhuanlan.zhihu.com/p/2057055806038844943)

This is a useful candidate for the missing sense of light spreading through thick lobes. The illustrated comparisons are appearance evidence, not a path-traced accuracy measurement. I found no reproducible GPU/resolution/pass-cost benchmark in the article or linked explanation that establishes our 1–2 ms target.

## Source observations

The published shader is a reading excerpt with HDRP stubs. Low-cloud density uses one base 3D lookup, one packed erosion lookup and several 2D lookups. A coarse mode omits erosion. Billowy and wispy densities use separate thresholds, and profiles vary with cloud type. Shadow work is shared with the diffuse term, but includes five additional weather-height reads per shaded point. Its low-cloud shadow range is capped at 6 km. [Shader](https://github.com/AshenOneArt/HPVolumeCloud/blob/main/VolumetricClouds.hlsl)

The difference from our up-to-13-lookup material is substantial at the algorithm level. It is not a measured speedup: texture bandwidth, arithmetic, occupancy and visual coverage still need profiling.

## How I would adapt it

**First priority: restructure our material and traversal.** Bake the fixed noise combinations, keep independent control over broad lobes and edge filaments, and move large-scale organization into weather/profile data. Avoid paying for our three warp samples, repeated octave combinations and detailed guides everywhere. Preserve the current angular-footprint filtering.

Use conservative occupancy to guide traversal rather than relying on an isolated coarse density sample as proof that a whole interval is empty. The bound must contain outward wisps and domain displacement. This matters both for silhouette detail and for stable guide ownership.

**Second priority: evaluate a diffuse illumination field in the shared lighting volume.** It should be a function of world position, illumination and cloud state, independent of the camera ray. Then primary rays and secondary misses can read the same field from their actual positions. Retain stochastic local shadow correction and fine density evaluation near visible folds. The result can be smooth in space without introducing an extra temporal denoiser before RR.

A possible experiment is to extend the light-cache payload with a scalar solar-response field and apply local atmospheric sun color at lookup. Where illumination color changes significantly across the field's support, especially twilight, use RGB or separately represented illumination instead. This is our proposed architecture, not an optimization demonstrated by the article. It must include distant shadow coverage so a local integration cutoff does not admit light through distant clouds.

Do not add another nested procedural light march to every `CumulusSource` call. With approximately **1.1 ms caches and 11 ms rays at 1080p DLAA**, that would expand the dominant cost. Budget the cache experiment separately and keep the previous lighting approximation available for controlled comparison.

**Third priority: validate scattering energy and scale.** Treat the new field as a candidate replacement for part of our existing higher-order contribution, with an explicit division between directional and diffuse scattering. Adding two estimates of the same scattering orders can brighten clouds twice. Keep the original homogeneous-step transport integral and single-scattering albedo convention.

## Physical approximations that need care

The author's explanation replaces full spatial transport and boundary conditions with fitted source/propagation factors. It also substitutes a slowly decaying absorption-survival factor for ordinary direct-beam attenuation. Those choices can produce useful images, but the displayed formula is not an exact consequence of the radiative transfer equation. [Author's derivation](https://github.com/AshenOneArt/HPVolumeCloud/blob/main/Docs/PhiFwd_FromRTE.md)

An isotropic angular field does not imply that the original droplet phase asymmetry can simply be discarded while retaining the same transport coefficients. Under the similarity approximation, use reduced scattering `sigma_s' = (1 - g) sigma_s` and reduced extinction `sigma_t' = sigma_a + sigma_s'`. In classical diffusion, `D = 1 / (3 sigma_t')` and the attenuation scale is `sqrt(sigma_a / D)`. PBRT explains the coefficient changes required when making the isotropic approximation. [PBRT diffusion treatment](https://pbr-book.org/3ed-2018/Light_Transport_II_Volume_Rendering/Subsurface_Scattering_Using_the_Diffusion_Equation)

For our tests, preserve optical thickness while scaling cloud size and extinction inversely. Mean brightness should remain consistent apart from deliberately changed atmospheric distances. Repeat with a refined integration step to expose step-size dependence. Use converged volume transport for a homogeneous slab and an isolated lobe before trusting a tuned daylight image. These are validation proposals, not tests performed in this analysis.

## RR and global weather implications

Keep density/depth/normal/albedo/motion guides tied to the actual cloud representation, not the diffuse-light field. A smooth light cache must not smooth away the geometric guides of detailed lobes. Keep the independent stable guide strategy, while reducing its cost through occupancy and local refinement.

Use planet-relative altitude, local up vectors and world-stable weather coordinates. Validate sun below the horizon, cloud undersides illuminated near sunrise, nearby reflections, camera motion and origin rebases. The daylight appearance of a local cloud field does not settle those cases.

**Decision:** adopt the cheaper material organization as a high-priority design reference, and prototype the diffuse-field idea inside our bounded lighting cache. It strengthens the previous optimization plan; it does not establish a measured shortcut from 12.1 ms to 1–2 ms.
