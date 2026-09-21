# Ocean audit and migration map — 2026-09-21

The user adopted OCEAN_OVERHAUL_AGENT_SPEC.md after an initial implementation.
The workspace already contained an uncommitted ocean implementation and unrelated
edits. No reset or checkout was used. The pre-compliance implementation is copied
to `out/ocean-spec/pre-compliance`; its image/timings are in `out/ocean-upgrade`.
That is an intermediate baseline, **not** an original-before-edit comparison.
The original untracked ocean has no complete frozen source/capture baseline.

## Repository evidence

| Area | Actual owner / symbol | Observed behavior and migration |
|---|---|---|
| Entry / API | `rdn/Main.cpp:MainScene`, `engine/Scene/Ocean.*:Init/SetParams` | Existing ship scene creates ocean through Renderer. Preserve these APIs and legacy fields. |
| Host | `rdn/ocean/OceanSystem.*:BeginFrame/record_gpu_work` | External stream on planet compute queue; four 1024² spectral cascades, global triangle buffers. Keep backend. |
| Units | `Ocean_Tiles_v8.hlsl:OceanTiles`, `append_instances` | Y up; metres; tile-relative positions translated into scene-relative world; world origin double on CPU. UVs retain undeformed tile coordinates. |
| Spectrum | `OceanCommon.h:Spectrum`, `OceanSystem.cpp:Bake` | JONSWAP, numerically normalized directional spread, polar-to-Cartesian Jacobian and bin area; independent cascade seeds. Legacy swell only narrows wind sea. Add independent swell and explicit height/period overrides. |
| Fourier | `Ocean_Sim_v8.hlsl:OceanEvolve/FftLine` | Positive, unnormalized inverse FFT and centred wavenumber lattice. Initial time sign propagated opposite to bearing. Test one mode, Hermitian symmetry, FFT/DFT and Parseval. |
| Derivatives | `OceanAssemble`, `OceanEvalSurface` | Eight real spectral fields available. Initial implementation inverted each band separately and ignored cross-band nonlinear deformation. Retain raw derivatives; compose before inversion. |
| Geometry | `OceanQuadtree.h:Select`, `OceanTiles` | Originally stitched 64² quad tiles with camera/frustum selection. Now 128² quads and complete finite-square ray coverage; finite extent and Earth curvature remain. |
| AS | `RecordAccelerationStructures`, `stream_orchestrator.cpp` | Per-slot BLAS updates/rebuilds; fixed scratch slots; graphics waits on compute. Audit previous/current field lifetime and actual geometry bounds. |
| Material | `MakeMaterial`, `ResolveSurfaceMaterial`, `Material_GGX_v8.hlsli` | Existing thick dielectric, diffuse/transmission weight and perceptual GGX roughness. Beauty selects the base water material with zero roughness; the reserved palette remains for compatibility and opaque diagnostics. |
| Optics | `Material_Common_v8.hlsli:CalculateAbsorptionThroughput`, `Material_Decoder_v8.hlsli`, `OceanOptics.hlsli` | Existing glass absorption retained. Water bypasses solid SSS and uses its controls for RGB free-flight sampling and one HG path event between object bounces. Water-only reflection proposal has a 70% floor with matching PDFs; other materials retain their original rule. Water membership follows surface crossings; camera-inside transport is supported. Nested media and multiple scattering are not implemented. |
| Footprints | `EvalSurfaceStateImpl`, `PtShadeHit` | Existing primary cone plus path distance/spread; explicit mip sampling. Tier B approximation, not exact focusing through refraction. |
| Foam | `OceanFoam`, `OceanMip` | Foam generation and shading removed at user request. Legacy entry point emits deformation diagnostics and zero coverage; reserved layout retained. |
| History | `SurfaceMotionVector`, `append_instances` | Generic instance motion cannot represent wave deformation. Add water-local previous displacement input through existing motion-vector interface. |
| Interaction | `engine/Scene/Ocean.h:SurfaceLevel` | Mean level API only. No ocean buoyancy, shoreline solver, collision/wake APIs found in these callers. Preserve curvature and mean-level behavior. |
| Configuration | `Params`, `MainScene::Init` | C++ parameters; no ocean serializer/editor panel present. Keep old fields; separate physical state and representation settings. |
| Execution | CMake/Ninja + MSVC; `include/dxc.exe`; test PowerShell scripts | Full application, SM6.6 simulation and SM6.9 render shaders run on this machine. GPU capture works. Runtime teardown did not finish promptly in initial trials. |
| Performance | `Core/PerformanceCapture.h`, ocean Stats | CSV graphics pass timing exists. Initial 20-frame sample is insufficient for steady-state claims. Compute timestamps and allocation accounting added. Prior 600-frame measurements predate the final detail/SSS changes; final performance unverified. User budget unknown. |

## Protected scope

Do not change `Pass_pt_trace_v8`, BSDF/Fresnel implementations, MIS, roulette,
`VisibilityTransmittance`, emitter handling, SHARC propagation, medium tracking,
path-state layouts, global exposure/lighting or denoiser settings. The initial
water-specific transport, shadow attenuation and guide attenuation were removed.
Permitted edits in shared files are limited to resolving water-local material
parameters, undeformed coordinates and water motion data. These are distinguished
from tracing/sampling changes in the final patch review.

The latest explicit user request additionally authorizes the water-only reflection
proposal change and pure-surface DLSS guides. `GGXReflectPick` now selects the water
proposal with matching evaluation PDFs; Fresnel and physical GGX response are not
changed. Water's SSS fields feed a material-layer approximation, not the solid walk.

The subsequent widened-sun experiment did not resolve the user's sparse glints
and has been removed. Latest direction explicitly requests mesh-light NEE and sun
NEE with a widened water lobe. The subsequent correction explicitly limits that
width to direct highlights: full-resolution normals and authored roughness now
drive continuation/guides again. Water NEE is taken in place on every eligible
vertex, uses a separate 0.14 minimum highlight roughness, and owns direct lighting
without a BSDF-hit MIS partner. A segment flag suppresses matching emitter/sun
hits while preserving ordinary sky and scene reflection. Thin-glass transmission
retains the flag; other surfaces reset it. Cache training matches this split.
This is an artistic direct/indirect response split. No packed layouts change.

## Latest user direction

The user owns all visual testing and explicitly requested no builds for the latest
revision. Earlier direction removed foam and kept intrinsic material roughness at
zero; the latest correction restricts the widened lobe to direct-light NEE only.
The subsequent explicit request authorizes thickness-dependent scattering and underwater volume transport, including the necessary trace, shadow and cache-training changes. Fresnel remains independent of the reflection proposal. See OceanRendering.md and OceanEvaluation.md for current changes and
unverified acceptance items. Earlier test/capture results do not certify this revision.

## Acceptance plan and known gaps

Add executed CPU/GPU mathematical checks mapped to T01–T29, with mathematical
references separate from production. Use the existing material GPU suite for
Fresnel/TIR checks; extend ocean tests for spectrum, deformation and foam. Record
unexecuted visual/interaction checks as such. Do not label an approximate GGX fit,
surface body-color closure or finite secondary-ray coverage exact.

Keep production light, exposure, sample count, bounce limits and reconstruction
settings unchanged in comparisons. Use fixed sea time/camera/seed for spatial
comparisons and deterministic time steps for motion sequences. No published
performance numbers are substitutes for measured local results.

## Consulted references

* [Tessendorf, 2004](https://people.computing.clemson.edu/~jtessen/reports/papers_files/coursenotes2004.pdf): Fourier waves, displacement and folding.
* [Horvath author implementation](https://github.com/blackencino/EncinoWaves): empirical directional spectrum conventions; no source copied.
* [LEAN author page and errata](https://userpages.cs.umbc.edu/olano/papers/lean/): filterable first/second slope moments.
* [Heitz, 2018](https://jcgt.org/published/0007/04/01/paper.pdf): GGX NDF; finite Gaussian covariance must not be called exact GGX variance.

