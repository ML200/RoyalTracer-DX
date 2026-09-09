# Video review: cloud sampling and moving-sun lighting

September 8, 2026. Input: `C:/Users/Malte/Videos/2026-09-08 12-02-18.mp4`, 1920x1080, 163 frames. The user confirmed advancing time and two lighting samples. Extracted frames show distant horizontal combing and broad red underside illumination changing while the cloud shapes remain nearly stationary. Review artifacts are under `out/cumulus/video-review`.

## Findings and implementation

The separate 48-step midpoint guide march introduced depth terraces independent of the 64-step noisy colour march. The new reconstruction shares deterministic linear density nodes between colour extinction and RR moments, computes depth within each interval, keeps fine front spacing on distant rays, and avoids integer step-count grid jumps. A fixed spatial offset decorrelates sampling planes. It leaves spatial error in raw silhouettes; it is not additional procedural detail or temporally convergent density noise.

A uniformly chosen density/lighting point could sit deep inside a thick interval yet supply the whole interval's outgoing light. Lighting now uses the optical median of reconstructed cell extinction. Smooth diffuse illumination is integrated at every occupied interval; only detailed direct shadows are selected sparsely. Direct-light selection accounts for incoming sunlight, with a nonzero cached-shadow importance floor. Its compensation preserves the discrete direct-light sum in expectation.

The distant primary shadow path previously switched abruptly from near-cache interpolation to a ten-sample procedural trace spanning hundreds of kilometres. It now blends the existing near and far caches and retains detailed local shadow samples. This improves coherence and cost, while accepting the far cache's spatial approximation.

The shadow trace's sample schedule formerly changed when the centre solar ray began hitting the planet. Fixed-distance strata now clip both cloud-shell intervals without moving the whole grid. Atmospheric and cloud direct illumination use the visible disk centroid for grazing transmittance/shadow directions, with disk fraction separately controlling energy. The finite-radius visibility width follows the geometric horizon factor in [Bruneton's reference implementation](https://ebruneton.github.io/precomputed_atmospheric_scattering/atmosphere/functions.glsl.html). The centroid approximation and fixed shadow schedule are implementation choices here, not claims of a full solar-disk integral.

The user's approved material and normal functions are unchanged. No texture allocations, bindings, history buffers, settings layout or executable changes were added.

## Verification

All checks pass: 1920x1080 GPU suite; ten-condition sparse-light convergence; dawn/dusk finite-disk and vertical-cache tests; atmospheric haze and zero-extinction controls; material/wind/silhouette tests; 17 production shader compilations. The zero-error after-sunset convergence case is now accepted explicitly rather than requiring a decrease from zero.

An independent physical-space CPU integration checks the GPU cell math: maximum inverse optical-depth residual 1.47e-6, maximum first-moment relative error 0.493%, second-moment error 3.502% in eight thin/thick/rising/falling cases. These bound the local quadrature checks, not the error of the entire procedural volume. Degenerate midnight direction, normalized visible-disk centroid, monotonic direction and above-horizon visibility checks pass.

The 65-position moving-sun sweep spans -0.9 to -2.5 degrees with 32 lighting selections per position. All 13 non-RGB channels remain identical. Normalized second differences of 12x8 cloud-patch mean radiance fell from 0.299 to 0.123 (59%). This diagnostic includes real changing shadows and is not an RR flicker score or proof of temporal stability under camera/wind motion.

Against a 512-interval reference, on reference clouds beyond 15 km:

| View | Opacity RMS before / after | Relative depth RMS before / after | Silhouette disagreement before / after |
|---|---|---|---|
| Ground | 0.094 / 0.066 | 0.026 / 0.015 | 1.52% / 0.98% |
| Cloud side | 0.039 / 0.004 | 0.282 / 0.183 | 0.13% / 0% |
| Above clouds | 0.244 / 0.167 | 0.279 / 0.163 | 6.80% / 3.53% |

These are reconstruction comparisons with unchanged density, not visual quality scores. Depth error remains significant on some long rays. Raw accumulated previews show sharper but spatially noisy silhouettes; RR's response needs in-engine validation.

At two lighting samples, 32-frame normalized raw RGB RMS on opaque cloud pixels fell from 0.087 to 0.016 in daylight, 0.762 to 0.129 in the ground sunset view, and 1.352 to 0.494 in the side sunset view. No denoiser was used. Fixed density quadrature removes temporal density noise, so these reductions do not mean spatial density error disappeared.

## Performance

Final paired RTX 5090 measurements, 1920x1080, two lighting samples, three alternating 32-frame batches per view. Milliseconds:

| View | Previous primary | Current primary | Previous cache | Current cache |
|---|---:|---:|---:|---:|
| Daylight ground | 6.60 | 4.88 | 0.81 | 0.49 |
| Sun-facing sunset, -1.3 degrees | 6.61 | 4.80 | 0.98 | 0.78 |
| After sunset, -3 degrees | 4.91 | 3.95 | 0.32 | 0.32 |

The primary pass improved about 20-27% in this final paired run. Absolute timings varied between runs: the preceding candidate measured around 4 ms primary, while its paired old version also ran faster. The final source includes the more accurate eight-point depth moments. Treat the paired log as the result, not a guaranteed in-engine time. Cache timing includes an ambient bake which the engine can reuse. The 3 ms total target remains unmet.

## Remaining limits and artifacts

This remains approximate single-ray volume transport with cached broad shadows, atmospheric hemisphere fill and five scattering octaves. It does not solve lateral cloud-to-cloud multiple scattering, a spectral atmosphere, or the full finite solar-disk visibility integral. The twilight appearance cannot be called path-tracing equivalent. Stationary per-pixel density offsets can leave spatial grain; the headless harness cannot establish its appearance after RR.

Rejected experiments are retained locally: random linear-density grids restored substantial sunset noise; increasing density samples everywhere cost roughly 6-8 ms and was not kept. The chosen revision removes the separate guide march and limits the extra primary intervals to distant entries.

Final artifacts: `out/cumulus/video-final` (GPU logs, moving-sun CSV/frames, guide reference/errors, integration checks, paired performance, raw noise, runtime hash manifest); `out/cumulus/video-production` (17 compiled shaders). Comparison programs are in `out/cumulus/video-review`. The pre-change shader backup is `out/cumulus/video-baseline/source`; the approved density snapshot and frozen reference tests remain intact.


## Temporal-noise follow-up

The user's subsequent screenshot exposed frozen density grain and atmospheric shadow slices. [CUMULUS_TEMPORAL_REVIEW.md](CUMULUS_TEMPORAL_REVIEW.md) documents temporally rotating colour/source quadrature, a separate stable RR guide pass, measured convergence and its added cost. Those measurements supersede the current-runtime claims above.
