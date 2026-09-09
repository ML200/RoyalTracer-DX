# SHaRC in the regular path tracer

SHaRC is enabled by default in the **Path tracer** integrator. Its controls are
in the integrator panel. ReSTIR skips all three cache passes and does not query
or update the cache. Disable **Enable radiance cache** for the original uncached
path tracer and its original bounce limits.

## Research and design

This is an original implementation, not a vendored NVIDIA SDK. The reference
design is NVIDIA's [SHaRC integration guide](https://github.com/NVIDIA-RTX/SHARC/blob/main/docs/Integration.md):
sparse update paths, temporal resolve, and a separate render/query pass.
Its current [hash grid](https://github.com/NVIDIA-RTX/SHARC/blob/main/include/HashGridCommon.h)
includes normal information in distance-adaptive world-space keys. Neither that
coarse normal code nor a hashed fingerprint alone proves that two hits belong
to the same surface.

This implementation extends that design with oriented geometric normals in the
key, smooth blending between signed normal axes, exact key verification, separate
surface records, and bilateral geometric/shading-normal and tangent-plane tests.
The use of surface descriptors and edge-aware support is informed by AMD's
[GI-1.0 report](https://gpuopen.com/download/publications/GPUOpen2022_GI1_0.pdf).
These are implementation choices, not a claim that this particular combination
is an established best method for every scene. Opposite blanket faces are
separated even when they share a voxel, instance and material. Nearby parallel
sheets can occupy separate records in the same spatial/normal bucket.

The path-footprint gate follows the spread-based reasoning in
[Real-time Neural Radiance Caching for Path Tracing](https://research.nvidia.com/publication/2021-06_real-time-neural-radiance-caching-path-tracing):
query only after scattering spreads the path enough to hide the spatial
approximation. A diffuse bounce number alone is insufficient around contacts
or behind sharp reflections.

## Passes and storage

1. **Prepare:** zero new/reset storage, reclaim entries past an adaptive lifetime,
   and evict levels that no longer fit their distance from the camera.
2. **Update:** run the same regular PT transport code at one rotating pixel per
   4x4 tile. Trace full training paths using the existing light tree, sun NEE,
   visibility, MIS, materials and media. Seed eligible primary and secondary
   surfaces. Never use cached radiance to terminate training.
3. **Resolve:** combine independent suffix observations with temporal history,
   update moments/confidence, and clear frame sums.
4. **Render:** run regular PT, retaining local NEE at each visited surface.
   Query the cache only for an eligible secondary continuation.

UAV barriers separate every producer/consumer. A single 160 MiB allocation holds
1,048,576 surface entries and survives resolution changes. The layout is shared
by HLSL and C++ in `shaders/SharcLayout.h`. The root signature uses 63 of the
available 64 DWORDs, including the three root UAVs.

Metadata publication uses a bounded compare/exchange claim. Contended insertions
retry on later training observations; no lane spins waiting for another lane to
release a metadata lock. Full buckets cause misses. Lookups and insertions search
past eviction holes and verify the complete signed XYZ/level/normal/instance/
material descriptor. Only update shaders request globally coherent memory access; rendering reads
an immutable snapshot behind a UAV barrier. Float accumulation uses atomic compare/exchange additions;
there is no fixed-point saturation or luminance clipping of rare bright samples.

## Open-world behavior

Grid nodes remain fixed in world space. Minimum spacing is 0.125 m; cell spacing
grows in powers of two according to distance from the camera. Training writes
both adjacent levels; queries blend those levels continuously. New camera regions
are populated by the rotating training pattern and its secondary paths. Stale
and obsolete detail is reclaimed, so walking through the world does not leave
the hash table permanently filled by the starting area.

Keys quantize the snapped scene origin and local hit position separately, using
integer cell coordinates plus a fractional local position. They never add a
planet-scale floating-point origin to a tiny local position before quantization.
The minimum usable level also increases if necessary to retain signed-coordinate
headroom. A camera origin rebase preserves history; geometry/material edits,
lighting-setting changes, mode re-entry, grid-spacing changes, and training-depth
changes reset it. Continuous sky motion is handled by rolling history. History decays on
frames that provide new observations to a cell, not on every elapsed frame.
Otherwise a fine cell visited once every 60 frames could never collect enough
effective observations to become useful. Base unused lifetime defaults to 512
frames and adapts to measured revisit intervals, capped at four times that setting.
Confidence fades over the latter half of that lifetime. Obsolete LOD eviction is
independent of the extended lifetime. Sparse histories consequently respond more
slowly to continuous lighting changes; explicit lighting edits still reset them.
Frequent scene edits conservatively reset the entire cache; local invalidation
for independently animated objects is not implemented.

## What is cached

Each record stores the albedo-demodulated outgoing **continuation after local
NEE**, not total radiance at that vertex. For a vertex `v`, the training label is
the BSDF-weighted suffix following `v`. It includes subsequent vertices' NEE and
the appropriate MIS-weighted emitter/environment contribution. The rendering
query occurs at precisely the same point, after adding `v`'s own NEE. This avoids
adding cached direct lighting on top of a second copy of local NEE.

Training propagates radiance backwards through up to eight registered vertices
using individual segment weights, including Russian-roulette compensation and
SSS throughput. Registration deposits the immutable surface descriptor into the
cache immediately; only two node handles, their splat weights, radiance and
throughput remain live across ray traces (320 bytes for eight vertices, formerly
800 bytes of vertex state). Demodulation is applied to the initial suffix weight. It does not divide by camera-prefix throughput. Each surface
receives one observation at path completion, including a zero observation when
the path dies. Repeated deposits into the same node within one path are not counted
again as independent observations.

Training defaults to a 48-vertex loop bound with roulette from depth 8 and 0.9
survival. Registration is limited to the first half of that path budget. This
leaves substantial transport behind each trained point. Sparse training cost is
independent of the render samples-per-pixel setting. Eligible primary hits also
seed the cache, so visible surfaces no longer depend entirely on secondary rays
finding them. Primary rendering still never terminates into the cache. Stochastic tent splats select
one interpolation corner and one signed normal axis per level, avoiding up to
48 deposits per vertex while retaining the same expected reconstruction kernel.

When SHaRC is enabled, cache misses use the same configured deep path limit and
bypass the ordinary diffuse-bounce cap. Otherwise a warm region could include deep
illumination while its cold neighbor stopped after three diffuse bounces. The
render path keeps its ordinary unbiased roulette policy. Like other radiance
caches, sharing finite training suffixes at different path depths remains an
approximation to long-path transport; it is not an unbiased finite-depth reference.

## Eligibility and stability

Cache termination requires all of the following:

- A secondary surface and a previously sampled diffuse scattering event whose
  accumulated footprint exceeds three coarse-cell widths (smoothly ramping to six).
- An opaque, non-SSS surface outside a medium, roughness at least 0.5, metalness
  at most 0.05, diffuse strategy probability at least 0.7, and negligible coat/sheen.
- Compatible geometric/shading normals, surface planes, view direction,
  roughness, material and reflectance.
- At least 16 effective observations by default, several updated frames, repeated positive
  evidence, low estimated relative error and recent training.

Trilinear position filtering, adjacent-level blending and continuous normal-axis
weights suppress grid/normal boundaries **in expectation**. The live query samples
one LOD, one signed normal axis and one trilinear corner using their reconstruction
weights, then scans that node's 16-slot collision bucket. This replaces up to 768
record probes with at most 16. Its expected accepted radiance and fallback
probability match the original deterministic interpolation, including partial
coverage. Individual samples carry additional Monte Carlo noise; temporal
accumulation/denoising is still needed. `SharcQueryLevel` retains the deterministic
reference for regression checks. Statistical confidence is computed at resolve
and read by queries, with an age fade applied at query time. Missing or rejected support lowers the
probability of using the cache; the remaining paths trace normally. Counts from
correlated interpolation neighbors are never added together. A further 1/32 of
otherwise eligible render continuations remain traced. Those render continuations
are not used for training; the separate update pass always supplies fresh evidence.

Zero-only history cannot certify darkness. A lone rare light discovery also cannot
immediately certify convergence. This deliberately favors tracing over a stable
but incorrect black cache or a persistent bright outlier. Hard scenes can therefore
warm slowly and may not gain performance. Glossy, layered, transmitting, grazing
and contact paths retain tracing. Albedo demodulation is approximate for the small
remaining specular component of eligible materials; view/roughness/reflectance
support constrains this error.

These gates address visible cells, but do not prove their invisibility under every
lighting distribution. In particular, same-facing surfaces closer than the plane
tolerance, unresolved sub-cell occlusion, and rapid lighting changes still warrant
scene-level inspection. This cache uses surface support rather than an additional
visibility ray between every interpolation sample and query point.

## Cache debug views

In the path tracer's **SHaRC indirect lighting** controls, set **Cache debug view**
to **Cells** or **Cell lighting**. The view automatically takes over presentation;
**Off** restores the previous display slice. Disabling SHaRC or selecting ReSTIR
also disables the inspector. Toggling the view does not reset the cache.

Both views project the nearest cache node onto camera-visible surfaces, at the
finer distance-adaptive level by default. **Show coarser cache level** selects the
other level used by the rendering query. They intentionally expose cell boundaries
and bypass spatial/LOD interpolation, denoising, sharpening and output dither.
They use render-resolution hits and nearest scaling at the display resolution.
Sky and terminal emitter hits use a dark blue background.

- **Cells:** stable colors from the exact spatial/normal/material key, with dark
  grid outlines. Dim colors indicate warming or uncertain history; brighter colors
  indicate higher statistical confidence. Missing compatible records are dark grey.
- **Cell lighting:** the resolved outgoing indirect continuation, remodulated with
  the stored representative reflectance, under the normal exposure and tonemap.
  This excludes local direct light and includes histories that are still too
  uncertain for rendering. Magenta means missing, amber means allocated without
  resolved samples, black means a stored zero, and red means non-finite data. Magenta/purple is a missing compatible record,
  not a NaN diagnostic.

The inspector searches all relevant signed normal bins, retains exact-key and
surface-plane checks, and deliberately ignores the camera/training-view difference
and rendering confidence/footprint gates. A colored cell is evidence of stored
coverage, not proof that a render path would terminate there. This is a surface
projection of one nearest node, not a volume view or the filtered final GI estimate.

Normal rendering and training continue underneath the view, preserving their
sampling, exposure and cache lifecycle. Inspection adds read-only lookups only
while enabled (one node rather than the full interpolation stencil); debug frame
times therefore include this overhead. Turning it off removes those lookups.
No buffer allocation or additional root constants are needed.

## Performance and warm-up diagnosis

See [non-ReSTIR performance work](NON_RESTIR_PERFORMANCE.md) for the current
dirty-mask resolve, material/pipeline changes, GPU measurements and validation.

The SHaRC panel shows **GPU ms: prepare / train / resolve / PT**, measured with
GPU timestamp queries. Results are read only after the existing previous-frame
fence, with no additional CPU wait. PT includes any enabled cache debug lookup.
These numbers separate actual cache costs from the older menu's CPU fence-wait
measurement labelled GPU. Debug views continue to add inspection work.

The standalone GPU runner also measures one million cold scattered queries and
one million warm surface queries. On the development GPU, before this revision,
the timings were 12.08 ms and 4.50 ms. The initial optimized measurements were
0.22 ms and 0.04 ms. These are synthetic **lookup-only** measurements, not a claim
about full-scene frame time. Deep training, visibility rays and full uncached
fallback paths must still be measured in the target scene. Enabling SHaRC also
continues to use the deeper cache path budget, so it is not a like-for-like bounce
budget comparison with the original three-diffuse-bounce uncached default.

Invalid training inputs are rejected before quantization or clamping. Resolve
replaces corrupt persistent history on the next valid observation, rather than
attempting a lerp through NaN. Query code skips invalid/rejected radiance before
multiplication, so `NaN * 0` cannot contaminate another cell's result.

## Verification and scene checks

Run `tests/run_sharc_tests.ps1` from a Visual Studio Developer PowerShell. It
compiles and runs the actual production cache/resolve/propagation code on D3D12.
The runner covers concurrent HDR accumulation, opposite faces 1 mm apart,
parallel sheets in one cell, eviction holes, fingerprint collisions, saturation,
stale eviction, black-history fallback, spatial/LOD/normal continuity, tiny
footprints, signed centimetre precision at 8,000 km across km rebases, and an
analytic 20-bounce roulette-compensated transport case. It also checks stochastic
reconstruction against a varying deterministic signal with partial coverage,
one training observation per 60 frames, adaptive sparse-cell retention, repair of
poisoned history and invalid-position rejection.

Compile the normal PT, update, prepare and resolve shaders with the project's
DXC options; rebuild `Pathtracer`. Shared shader changes also require compiling
the ReSTIR/camera/shading shaders. GPU unit tests establish numerical and memory
behavior, not scene appearance or a frame-time improvement.

For visual acceptance, compare SHaRC against a converged uncached reference with
the same deep bounce budget: a thin blanket over a bed, parallel folded cloth,
an interior lit only through several diffuse bounces, camera movement through
1 km origin changes, a teleport into a cold region, and light/material edits.
Inspect raw output as well as DLSS output. Measure cold and warm frame times
separately; lower training tile width improves refill at additional tracing cost.
