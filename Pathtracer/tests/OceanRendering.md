# Ocean implementation and review controls

Latest revision: source only. The user builds and performs all visual testing.
Do not treat earlier captures or successful test runs as validation of this revision.

## Current changes

* Water takes mesh-emitter NEE and sun NEE once at each outside-facing water vertex,
  independently of its continuation lobe pick. Only these direct-light evaluations
  use `max(authored roughness, 0.14)` via `OceanHighlightRoughness`. Sky, scene
  reflections, refraction and DLSS guides retain authored roughness and full-resolution
  wave normals. No averaged normal or footprint roughness enters continuation.
* Direct water highlights use NEE alone (weight 1), not MIS against the differently
  shaped continuation lobe. An outgoing upper-hemisphere segment suppresses immediate
  sun/emitter-hit radiance already covered by NEE. Ordinary sky radiance remains.
  Mesh-hit emission remains enabled when mesh NEE is disabled. The marker survives
  thin-glass transmission and resets at other surfaces; refraction into water keeps
  its ordinary transport. Cache training follows the same rules.
* Water direct lighting is resolved in place and excluded from primary diffuse reuse
  and water-surface cache shortcuts. Other surfaces keep their deferred/MIS path.
* The water-only delta shortcut is removed: sampling formerly fixed H=N below
  roughness 0.06 while evaluation returned a finite GGX lobe/PDF. Water now samples
  the VNDF used by that evaluation. Its existing numerical alpha minimum remains
  0.001; no new constant roughness floor was added. Other materials retain their
  sampler. The unsuccessful enlarged-sun workaround and its flags/reuse exclusion
  have been removed; the original solar disc is restored.
* Foam is removed from generation and beauty shading. Legacy foam parameters and
  reserved descriptors remain compatible but cannot enable foam. The legacy
  `OceanFoam` entry point now only publishes deformation diagnostics, with zero
  foam channels. Its timing column still uses the old name.
* Authored water roughness remains zero. The full-response normal-distribution filter
  was removed after the user reported loss of glossy reflection. Large waves and
  mesh displacement are unchanged. Global exposure, sample
  counts and reconstruction settings are unchanged.
* Wave textures now use 1024 x 1024 samples per cascade (four times the texels).
  The periods are 8192, 977, 119 and 17 m. Restoring the final 17 m patch increases
  the available metre-scale wave modes; it has ~16.6 mm sample spacing. The 119 m
  patch now has ~116 mm spacing. The 128 x 128 quad tiles and 8 m minimum setting remain.
* Short wind-wave amplitude ramps from unchanged above 8 m to a 1.5 gain below 2 m.
  This is an explicit artistic control, not a new measured ocean-spectrum claim.
  Independent swell is unchanged. Mean-level/height prediction includes this gain.
* Horizontal crest sharpening rolls off for ripples below 0.5 m; their vertical
  heights and gradients remain. These ripples no longer dominate the deformation
  limit applied to the entire sea. The bounded composite horizontal strain is now
  0.85, retaining a minimum stretch of 0.15 (previously 0.35). No folding is allowed.
* The water dielectric now samples reflection with a minimum 70% probability,
  increasing with Fresnel when its reflection share exceeds that floor. All GGX
  PDFs use the same proposal; physical Fresnel and BSDF values remain unchanged.
  This is the reflection/refraction split within the dielectric lobe. Authored diffuse,
  sheen and coat have separate lobe selection. Ordinary glass keeps its old rule.
* Water DLSS guides always describe the primary surface: depth, normal, roughness,
  material albedos and deformation motion. Both motion guides follow the waves;
  specular hit distance is zero. Water bypasses mirror-chain/background probes.
* Water's Subsurface controls describe a homogeneous participating medium. Each
  underwater ray samples an RGB-mixture free-flight distance against its actual
  nearest geometry (including a miss). A collision redirects that same path once
  using the HG phase function; otherwise it reaches the surface. Both branches
  divide by their sampling probabilities. Extinction applies after the one event.
  Object bounces and fresh water entries reset the allowance; internal reflection
  and thin-pane transmission preserve it. Actual surface Fresnel governs exits.
* Camera classification runs once in the camera pass and is stored in an existing
  flag word. Underwater paths start at the lens, including camera misses/direct
  emitters, while the original surface record remains for denoiser guides. The
  final pass no longer integrates a second volume or adds underwater atmospheric
  aerial perspective. Cache training uses the same free flight and reset rules.
* The previous two-point sun/sky integral is removed. Volume evaluation now makes
  no height queries or shadow connections: each sampled collision adds one normal
  continuation trace. The surface-bounce limit stays unchanged, with at most one
  added collision between object bounces. The default material coefficients,
  accepted direct-highlight width, sharp reflection and Fresnel proposal remain.

The 1024-square fields increase texture memory and simulation cost; neither cost nor
visual convergence has been measured for this revision. No build or visual test was run.

## Parameters and conventions

`ocean::Params` is in `rdn/ocean/OceanCommon.h`; the scene configures it in
`rdn/Main.cpp`. Public `Ocean::Init`, `SetParams` and `SurfaceLevel` remain available.
The last API returns mean sea level, not a dynamic buoyancy query.

World units are metres, Y is up, and bearings are clockwise from +Z. The FFT uses
positive, unnormalized spatial synthesis and exp(-i omega t) traveling evolution.
Initial complex Gaussian amplitudes use 0.5 sqrt(PSD * bin area); cascade power
weights partition unity and amplitudes use their square roots. DC and ambiguous
Nyquist row/column modes are zero. CPU time is double precision; 128-second epochs
are folded into coefficients without reseeding.

`significantHeight` and `peakPeriod` override wind/fetch when nonnegative/positive.
Negative defaults retain legacy wind/fetch behavior. `swellHeight`, `swellPeriod`,
`swellDirectionDeg`, and `swellSpreadDeg` describe an independent swell component;
legacy `swell` still controls wind spreading. `amplitudeScale` is global artistic gain;
`shortWaveAmplitude` (default 1.5, range 0..3) adjusts short wind waves only.

`choppiness` is the requested horizontal gain. Each frame the simulation reduces
maximum secant/analytic strain bounds before applying a uniform safe gain. The
sum of band bounds limits the composite horizontal map; it does not clip vertices.
GPU readback reports `conditioningGain`.

`chlorophyll` is mg/m³; `turbidity` scales particulate scattering.
`legacySubsurface` is retained for source compatibility but has no effect.
`subsurfaceStrength` scales scattering density (zero disables scattering only);
`subsurfaceRadiusScale` scales the derived scattering mean free path. The editable
material uses sigma_s = scatteringColor * density / meanFreePath, and sigma_a = Tf
in inverse metres. `subsurfacePhaseG` is HG anisotropy. The preset uses the existing
chlorophyll/turbidity coefficients without an artistic green multiplier. Foam
controls are inactive. `bodyWeight` is optional surface opacity, default zero.

`minTileSize`, `lodFactor` and `extent` control geometry. `filterScale` now affects
footprint diagnostics only. The direct-highlight width is local to `OceanOptics.hlsli`.
The complete finite square remains ray visible in every direction. `nearKeepRadius`
is retained for source compatibility but no longer culls off-screen geometry.
Fields use undeformed barycentric coordinates and a texel-centre correction.
Geometry and shading sum displacement derivatives before nonlinear inversion.
Respecification invalidates displacement history; ordinary camera motion does not. Previous displacement follows the current
stitched mesh topology; LOD-change correspondence remains an approximation.

## Limits of the existing closures

The solid-object SSS walk remains excluded. This is single scattering between
object interactions, not unrestricted multiple scattering: after the one event,
additional scattering is extinction rather than another bounce. Object bounces
restart the allowance. There is no additive green layer or ad hoc energy boost.
The RGB free-flight estimator uses sigma_t=sigma_a+sigma_s; its event density is
mean(sigma_t*Tr), with weight sigma_s*Tr/pdf. The probability of reaching geometry
is mean(Tr), with weight Tr/pdf. An absorption-only medium attenuates deterministically.
HG phase sampling cancels phase value/PDF. Surface Fresnel and the 70% reflection
proposal remain separate from volume sampling.

Volume rays now reach actual objects, emitters and wave interfaces. They have no
volume NEE, so small bright sources can have high variance. The previous planar
sun/sky connections are gone. Surface-light shadow connections still use the
existing approximate local height-plane water interval. Camera initialization
uses the FFT height field, with actual triangle-side correction for a directly
visible water interface. Nested glass/air cavities still need a general medium
stack. Finite XZ extent is clipped. Runtime, noise and denoiser quality are unmeasured.
Shading normals represent fine-wave orientation, not displaced silhouette or
self-occlusion below the mesh resolution. Convergence at subpixel ripple scales
is not guaranteed at low sample counts. The direct-only highlight lobe is an
intentional artistic approximation: direct lights and reflected scene/sky radiance
use different responses. It is not an unbiased simulation of one physical BSDF.
NEE retains actual light geometry and visibility, and adds shadow rays; current GPU
cost is unmeasured. No claim is made about arbitrary refractive light chains.
Water's specular motion follows the deforming
surface; exact motion of reflected glints and long mirror chains is not claimed.
Favoring reflection samples leaves fewer transmission samples and can increase
variance in underwater contributions. Denoising improvement needs user validation.

## User build and visual review

The Materials panel lists **Water** first and hides reserved diagnostic variants.
Its ordinary surface, roughness, IOR, transmission and SSS controls edit the live
water material. The initial roughness remains zero. An editor change takes ownership
of the material for the current scene, so per-frame ocean defaults cannot overwrite
it. These are live session edits, not a new save/persistence feature. Procedural wave
normals remain active; material overrides are retained only in ocean debug views.
Under **Water > Subsurface**, use **Water body scattering**, **Scattering color**,
**Scattering depth (m)**, **Forward scattering** and **Body strength**. Disable the
checkbox or set Body strength to zero to compare against the clear dielectric.

Build the normal `Pathtracer` target and restart the renderer. The shader assets
and executable must come from the same build because shared layouts changed.
The application has not been built or launched for the latest revision.

Optional scene fixtures: `RT_OCEAN_FIXTURE=calm`, `mixed`, `rough`, or `flat`.
They use a fixed 1/60 s simulation step and leave lighting/tracing settings alone.
`RT_OCEAN_PAUSE=1` freezes waves. Unset it to review motion vectors.
`RT_OCEAN_DEBUG=1..6` selects material diagnostics: normal, compression, covariance,
zero footprint roughness, removed foam (black), and diagnostic footprint mip. These colors are lit diagnostic materials,
not raw numerical readbacks. Unset it for beauty. Existing motion-vector views can
now inspect the water guide outputs with a stationary camera.

Tests, when the user elects to run them from a VS developer shell:

```powershell
./tests/run_ocean_tests.ps1
./tests/run_sharc_tests.ps1
./tests/run_render_pipeline_tests.ps1
./tests/run_surface_precision_tests.ps1
```

The first uses production CPU spectra/quadtree and production GPU FFT, derivatives,
conditioning and mip kernels against a double-precision DFT on a small lattice.
The material suite retains deformation, absorption, Fresnel, Snell and TIR checks.
Added water checks exercise actual smooth GGX reflection frequency, TIR, unchanged
BSDF values with the changed PDFs, and bounded/depth-responsive body return.
New checks compare the sharp continuation peak and the wider off-mirror NEE response,
direct-light ownership by hemisphere, non-delta water sample spread and the GGX
normal-incidence slope CDF. PDF comparisons include zero authored roughness.
Current GPU tests have not been run.

## References

Epic separates surface meshing from water scattering/absorption controls. Its
Single Layer Water documentation describes a custom raster volume/compositing
path, not a replacement for this renderer's path tracer. Used as conceptual
reference only; no Unreal source was copied and the screenshot's exact material
cannot be inferred from the image.

* [Epic: water meshing and tessellation](https://dev.epicgames.com/documentation/unreal-engine/water-meshing-system-and-surface-rendering-in-unreal-engine)
* [Epic: Single Layer Water](https://dev.epicgames.com/documentation/unreal-engine/single-layer-water-shading-model-in-unreal-engine)
* Earlier Fourier/statistics references are in `OceanAudit.md`.
* [PBRT: importance sampling](https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/Importance_Sampling)
* [NVIDIA: Filtering Distributions of Normals for Shading Antialiasing](https://research.nvidia.com/publication/2016-06_filtering-distributions-normals-shading-antialiasing)
* [NVIDIA: DLSS Ray Reconstruction guides](https://github.com/NVIDIA-RTX/Streamline/blob/main/docs/ProgrammingGuideDLSS_RR.md)
