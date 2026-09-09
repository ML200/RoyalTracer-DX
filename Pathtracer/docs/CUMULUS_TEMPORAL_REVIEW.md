# Temporal density and atmospheric sampling

September 8, 2026. Follow-up to the user-approved lighting/shape revision: static grain at cloud edges and visible horizontal levels in the atmosphere.

## Changes

Cloud colour now rotates its linear-density integration grid with the existing spatially decorrelated low-discrepancy frame phase. It changes the quadrature, not the density material or lighting model. The old fixed per-pixel offset left a spatial error that could not converge over time.

RR guide opacity, depth, depth spread, normal and motion retain a separate fixed density reconstruction. `Pass_cumulus_guides_v8.hlsl` runs after the primary atmosphere pass and before shading. The primary pass clears guide outputs, including when underground; disabled clouds skip the guide pass. Both passes are included in the Cloud rays GPU timer. Sharp secondary misses retain stable virtual-hit guides and cheap colour integration.

An initial combined colour/guide kernel was too expensive. Splitting them lets the driver compile the guide kernel independently. Its compiled resource list contains only the camera/constants, density textures and output; no atmosphere or lighting textures survive dead-code elimination. This fixes the static noise at a cost: a separate density integration is required for stable guides.

Clear-air source samples also rotate each frame. Each fixed extinction interval samples an optical distance using the least-extinguished RGB channel as a proposal, then applies RGB compensation. This replaces the fixed midpoint shadow slices. Extinction remains evaluated at a fixed midpoint, so changing source samples cannot change the clear-air camera transmittance. Air source density, solar visibility, cloud shadows and atmosphere multiple scattering are evaluated at the sampled position. Deterministic LUT bakes keep the original sampling path. The piecewise extinction approximation itself remains.

## Validation

The executable builds successfully. All 18 production shader entry/variant compilations pass. The 1920x1080 GPU suite, lighting convergence, twilight/cache tests, atmosphere controls and analytic integration probes pass.

The new temporal regression isolates density by using full lighting evaluation: mean edge-opacity change across frames is 0.0427, with all nine guide channels bit-identical. This test fails a frozen density grid even when direct shadow sampling still creates RGB noise. Lighting-budget comparisons continue to preserve all 13 non-RGB channels for a fixed density/source phase.

For a cloud-shadowed haze view at 3.5 km, the relative RGB error against a 64-step reference was 3.95% for fixed 12-step midpoint sources, 5.13% for one stochastic frame, and 0.94% after 256 temporal samples. Source jitter left air extinction bit-identical. The independent homogeneous-medium probe matches the analytic RGB source integral within 8.48e-6 relative error across eight optical-depth cases. The haze reference still approximates heterogeneous extinction; this is not a comparison against full volume path tracing.

Accumulated raw previews remove the fixed grain from the cloud silhouettes. The haze comparison shows the fixed levels smoothing toward the denser reference. These are headless results without RR; final in-engine denoiser behavior still needs manual review.

## Performance and artifacts

Stable guides add a separate density pass. Current paired 1080p measurements and the final runtime manifest are under `out/cumulus/temporal-final`. An initial split-pass measurement was around 6.4-6.7 ms for primary colour plus guides, versus 4.1-4.2 ms for the preceding fixed-noise version; after-sunset cost was about 5.9 ms versus 3.4 ms. Cache timing was similar. This is a correctness/denoising revision, not a performance improvement; the 3 ms total goal remains unmet.

No shader samples or cloud detail were reduced to hide the extra cost. Future optimization should preserve independent stable guides and temporally varying colour rather than returning to frozen noise. The density/normal material functions and the accepted lighting model remain unchanged.

Artifacts: `out/cumulus/temporal-final` contains the app build, GPU/regression logs, haze images, source/runtime manifest and paired timing logs. `out/cumulus/temporal-production` contains the 18 production compilations. `out/cumulus/temporal-baseline/source` preserves the preceding version; `temporal-v1` retains the rejected combined kernel. The frozen historical reference files were not modified.

Final isolated paired run, three alternating 32-frame batches at 1920x1080; primary time includes colour and stable guides, milliseconds:

| View | Previous primary | Current primary | Previous cache | Current cache |
|---|---:|---:|---:|---:|
| Day | 5.18 | 8.30 | 0.49 | 0.49 |
| Sunset | 5.17 | 8.54 | 0.79 | 0.79 |
| After sunset | 3.80 | 6.47 | 0.32 | 0.32 |

These headless GPU times exclude RR itself. Cache time includes the ambient bake, which the engine can reuse.
