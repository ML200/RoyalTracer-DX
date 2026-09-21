# Ocean evaluation status

Latest feedback revision: **not built or visually tested**, at the user's explicit
request. The costly two-sample sun/sky volume integral is replaced by one stochastic
path-scattering event between object bounces. Camera classification is reused;
camera-origin water rays use the path loop. A phase sample follows real geometry
and interfaces, with no volume shadow rays or height solves. Surface reflection,
highlight width, Fresnel/proposal separation and material coefficients are unchanged.
`OceanVolumeReference.ps1` passed 89 scalar checks and four source contracts for
RGB free-flight event and endpoint energy, absorption-only/spent-event limits,
HG moments, Fresnel independence, event reset and camera/volume wiring. GPU tests
were updated but not run. Build, speed, noise and denoiser acceptance belong to the user.
Earlier 170-check integration results describe the removed volume estimator.
Previous direct-only revision: whitespace/diff check passed. Production highlight
width/hemisphere helpers passed 1,011 scalar JS checks. Nine source contracts
checked in-place NEE ownership, sun/emitter-hit suppression, the mesh-NEE-disabled
fallback, segment forwarding and matching cache-training handling. Full-resolution
normal sampling and absence of footprint roughness in material fetching were
checked. A scalar GGX comparison confirms the continuation peak stays sharper
and only the direct-light lobe gains off-mirror response. No compilation, GPU test
or renderer launch was performed; visual acceptance remains with the user.

Historical, removed full-response filter verification: whitespace/diff check passed. Independent
double-precision normal-Jacobian checks over 2,000 random deformations/covariances
matched the angular-variance expression with maximum relative error 4.80e-10
(absolute 1.73e-9). Added GPU checks cover off-mirror sampling and the GGX slope
CDF at zero authored water roughness, covariance propagation, and material PDFs.
Those GPU checks, emitter-size comparisons and all visual checks are unexecuted.

Previous sampling/body-layer non-build verification: whitespace/diff check passed. A component-wise JS
evaluation of the production `OceanOptics.hlsli` helper expressions passed 7,481
scalar checks for proposal-weighted means, bounded/depth-monotone body return and
energy partition. Maximum weighted-mean error was 1.78e-15. This does not validate
shader compilation, GPU sampling or DLSS reconstruction; the new GPU tests remain
unexecuted.

Historical, removed solar-filter experiment: production scalar cap and sun-state helper
code evaluated in JS passed 18 state cases, including inactive/non-water identity,
zero radius and a capped large disc. Solid-angle-integrated radiance error was
zero; maximum PDF normalization error was 2.88e-11. A 20,000-bin integration of
the RGB limb profile preserved each channel's energy. The default radius changes
from 0.265 to approximately 0.530 degrees, area increases fourfold and peak
radiance becomes one quarter. These checks do not measure rendered hit frequency,
occlusion, frame timing or denoiser stability. No build or GPU test was run.

Base repository revision: b093d7bad4edd29879bfc0bb2f5bfa4f89556841, plus the user's
pre-existing local work and these uncommitted changes. The initial ocean was
untracked; no complete original-before-edit capture/source baseline exists.

## Evidence and status

| Tests | Status for current source |
|---|---|
| T01 zero sea, T02 travel, T03 FFT/DFT, T04 Hermitian, T05 Parseval, T06 derivatives | Production-kernel numerical harness exists. Earlier revision passed FFT relative L2 4.35104e-7 and Parseval 4.09759e-7. Latest revision not rerun. Single-mode crest tracking is not a separate executed fixture. |
| T07 normalization, T08 cascade/resolution, T09 direction | Earlier production CPU spectrum/partition tests passed: ensemble ratio 1.00331, relative SE 0.00282122; direction error 0.000177569. Latest 1024-square fields, restored 17 m cascade and short-wave gain not rerun. Resolution invariance render test not run. |
| T10 composite normals, T11 conditioning | Earlier reference finite-difference max error 3.71789e-8. GPU stress applied gain 0.134325 and verified minimum stretch >=0.35. New frequency-dependent horizontal chop and 0.15 stretch margin have source/reference checks only; updated production-kernel suite not run. |
| T12 ray-visible waves, T13 hit coordinates, T14 seams | Shared displacement, UV and stitched geometry code reviewed. Earlier general surface precision suite passed. Dedicated ocean primary/secondary ray and LOD seam tests not run. |
| T15 origin rebase | Earlier phase arithmetic check passed. Latest motion fix removes duplicate rebase; runtime rebase validation belongs to user. |
| T16 moments, T17 covariance | Earlier production raw moment/mip test passed. Full tilted/high-deformation covariance stress suite not run. |
| T18 representation radiance, T19 anisotropy | Authored roughness zero; full-resolution normals restored. Wider direct highlights are evaluated by NEE alone; scene/sky continuation stays sharp. This direct/indirect split is intentionally approximate. Updated GPU regressions and visual acceptance remain unexecuted. |
| T20 Fresnel, T21 absorption | Water-only 70% reflection proposal floor with matching PDFs; physical Fresnel unchanged. RGB free-flight sampling and one HG path event per object bounce replace the body layer and the intermediate sun/sky integral. 89 scalar checks and four source contracts passed; GPU tests are authored but unrun. Multiple scattering, focused caustics and a nested-medium stack remain unsupported. |
| T22 foam timestep, T23 transport/history, T24 coverage filtering | Superseded by explicit user request to remove foam. Foam generation/shading and obsolete helper tests removed; legacy simulation entry only emits deformation diagnostics. |
| T25 temporal data | Water has a dedicated surface-only guide path and skips background/mirror probes; source review only. Deformation-vector and specular reconstruction visual acceptance pending user. |
| T26 legacy API | Public Ocean API retained. New radius/phase controls added. No dynamic buoyancy/wake/shoreline API was found in the existing callers. Runtime compatibility not exhaustively tested. |
| T27 lifecycle | Earlier driver compilation crash disappeared after reducing shader expansion and moving motion evaluation out of raygen. Latest build and device/resize/reload stress not run. |
| T28 non-water regression | Earlier material/SHARC, pipeline and precision suites passed. New material decoding and GGX proposal branches are water-only; solid SSS unchanged. Current source not rerun. |
| T29 performance | Budget not specified. Latest 1024-square simulation costs unmeasured; legacy SSS disabled by default. Earlier timing is not current-performance validation. |

## Pipeline creation regression

The user reported DXGI_ERROR_DRIVER_INTERNAL_ERROR (0x887A0020) while loading the
pipeline, before rendering; DRED has no breadcrumb nodes. No runtime GPU hang or
faulting source line is established by that report. The new training retrace loop
carried and replaced an opaque HitObject; this is the leading source-level suspect,
not a confirmed driver diagnosis. Both path loops now snapshot trace results into
ordinary IDs, barycentrics and distance immediately. Reordering uses integer hints
at completed endpoints, so no HitObject survives a retrace or reorder boundary.
Traversal flags, alpha handling, scattering weights and material behavior are retained.
Pipeline logs now bracket compute PSO creation and ray-tracing state-object creation.
The workaround is source-verified only: 89 scalar checks and five source contracts
pass. No build, shader compilation or runtime test was performed. User rebuild must
confirm whether state-object creation succeeds; this is not yet a verified crash fix.
## Historical measurements, not current acceptance

Before the user reserved visual testing, the mixed fixture rendered 725 frames on
an NVIDIA RTX 5090, driver 610.88, at 1920x1080. Frames 120–719 supplied 600 samples.
The scene used wind 11 m/s, Hm0 2.5 m, Tp 7 s, swell Hm0 1.7 m / 11 s, seed 1337,
dt 1/60 s; default scene camera and unchanged production lighting/tracing controls.
It used the earlier conservative limiter, 64² geometry, no SSS binding, and the
17 m finest band. No matched original baseline exists. Captures and logs remain
in `out/ocean-spec`; **they do not show the current revision**.

| Earlier stage | Mean ms | p95 ms |
|---|---:|---:|
| FFT/history | 0.1005 | 0.1026 |
| Assemble | 0.0212 | 0.0223 |
| Foam | 0.0202 | 0.0214 |
| Mips | 0.0903 | 0.0921 |
| Mesh | 0.0419 | 0.0424 |
| Ocean BLAS | 1.5959 | 1.8540 |
| Ocean compute total | 1.8700 | 2.1279 |
| Graphics GPU frame | 5.8944 | 6.9701 |

Earlier allocation accounting: 588,523,520 bytes including ocean resources and
reserved global geometry, excluding global reconstruction buffers. It is obsolete
for the new 128² tiles. No final memory or real-time budget pass is claimed.

Reproduction commands and review controls are in `OceanRendering.md`.
