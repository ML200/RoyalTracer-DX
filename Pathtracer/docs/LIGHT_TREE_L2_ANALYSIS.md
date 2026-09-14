# Light-tree locality analysis — 2026-09-10

The strongest measured opportunity is coherent traversal among nearby receivers. Compact CDF storage alone did not materially improve stable sampling in the large-light fixture. A screen-space sample cache is plausible, but copying one pixel's light/PDF to nearby pixels would require changes to the estimator and emitter MIS. There are lower-risk ways to improve coherence first.

This pass changes no production renderer code. Experiments live under ignored `out/light-tree-research/l2-analysis`. Measurements below are D3D12 GPU timestamps on the RTX 5090, not per-resource L2 counter captures or full-scene image tests.

## What the displayed PT timer contains

`rdn/Renderer.cpp:2440` opens the PT timer at `Pass_pt_nee_v8.hlsl` and closes it at `Pass_pt_v8.hlsl`. Thus the single displayed PT number includes both primary NEE prefetch and the bounce kernel.

Primary prefetch already runs in a separate 16×16 compute shader (`shaders/Pass_pt_nee_v8.hlsl:18`). It selects a triangle independently for each eligible primary pixel and stores its selection PDF, RNG continuation and learning tokens. The bounce kernel selects the point on that triangle and traces visibility. Additional initial samples, secondary vertices and cache-training paths still sample inline (`shaders/Pass_pt_v8.hlsl:443`). Emitter MIS evaluates a light-tree PDF at the previous receiver (`:867`).

A primary screen cache can reduce the displayed PT time. It cannot by itself organize secondary receivers: neighboring camera pixels may hit unrelated positions, normals and materials after a bounce.

## Measured experiments

Fixture: 262,144 equal-power triangle lights, one BLAS, 512 receivers, 2,073,600 queries, and approximately 31 learned clusters per receiver after 138 warmup updates. Coherent input groups 256 consecutive queries at one receiver; scattered input hashes each query to a receiver. These are deliberately controlled locality cases, not a captured camera/G-buffer. Each timing is the median of five dispatches.

Dedicated sampling kernels were used for the final comparison, so unrelated regression-test branches do not determine their register footprint. All four variants passed the existing GPU regressions, including sample/MIS agreement, movement, learning and exact feedback accumulation. The correlated variant only enables shared randomness for the benchmark; those regressions do not establish its spatial image quality.

| Stable-LOD sampling only | Current | Packed CDF | Four-lane correlated tree RNG | Both |
| --- | ---: | ---: | ---: | ---: |
| Coherent receivers | 1.209 ms | 1.204 ms | 0.250 ms | 0.248 ms |
| Scattered receivers | 1.456 ms | 1.470 ms | 1.287 ms | 1.241 ms |

Packing the CDF alone was essentially neutral here. Four-lane correlation made coherent sampling approximately 4.8× faster, while scattered sampling improved only about 12%. Every lane still executed its own traversal with its own receiver and evaluated its own PDF; no sample-result cache was involved. Matching random choices lets coherent lanes follow the same branches and reuse node data. It also correlates their light samples, which must be assessed visually at equal time.

The large tree exposed much more expensive PDF and LOD-mixture paths than the earlier 128-light atomic benchmark. Timing spread and shader differences make those results unsuitable for attributing a precise speedup to CDF packing. The stable sampling-only comparison above is the useful controlled result. Raw final logs are `<variant>/dedicated-benchmark.log`; `benchmark.log` contains the earlier general-test-kernel measurements and should not be used for the final comparison.

## Memory and access findings

### Learned cuts mix read-only proposals with written counters

`LTC_SampleCell` binary-searches CDF values at `cluster + 44`, with a 64-byte record stride. A 32-entry CDF contains 128 bytes of useful values spread across a 2 KiB record region. On a 32-byte-sector model, its values occupy 32 sectors instead of four when packed. This is an access-footprint calculation, not a measured cache-miss ratio.

The selected cluster then supplies node/BLAS IDs, trail/depth and probability. Its live feedback count at `+56` shares a 32-byte sector with frozen probability/CDF at `+40/+44`. Moving the wide sums out of the record removed the retry storm, but did not fully separate immutable sampling data from writes.

A production redesign should separate all mutable statistics, including this count. A 32-byte frozen record can hold node, BLAS, trail, depth, probability and CDF, reducing the complete frozen-record plane from approximately 64 MiB to 32 MiB. A dense CDF or alias plane can additionally reduce scalar selection traffic. The experimental packed CDF merely added a 4 MiB copy and changed the CDF reads; it did not implement this complete separation. Its neutral result does not evaluate the full redesign.

The current 297.03 MiB learning allocation is not all read by sampling. Its accumulator plane receives feedback and is consumed during prepare; treating allocation size as the sampling working set would exaggerate capacity pressure. Separate resources for immutable proposals and writable feedback would also permit explicit SRV/UAV usage; actual cache-policy benefits should be verified in the generated shader/profile.

### Tree descent remains dependent and divergent

TLAS and BLAS nodes are 64 bytes. Each internal level reads up to four contiguous children and chooses the next node. Sibling adjacency and carrying selected-child topology in registers are already implemented. Random choices send neighboring lanes into different subtrees deeper in the tree, making their child blocks scattered. Reordering screen pixels cannot prevent that on its own.

The same structure is traversed again for emitter PDFs. During an LOD crossfade, `LT_SampleLight` samples one proposal and evaluates the other proposal's PDF for the same light (`shaders/LightTreeLearning_v8.hlsli:276`). The extra cut lookup and subtree PDF are real work on every blended NEE sample. Compacting a five-step CDF search cannot remove these traversals.

A 32-byte conservatively compressed tree node is a separate worthwhile prototype: parent-relative bounds, packed cone data and topology. Bounds and cone quantization must expand outward, and both sampling and PDF evaluation must consume the same representation. Large-world coordinates should not simply be converted to half precision. A smaller record also trades decode arithmetic and registers against traffic; measure both.

### Approximately half the uploaded fixture nodes are unreachable placeholders

The recursive builder reserves sibling slots, separately appends each child root, then swaps the root into its reserved slot. The swapped-out placeholder remains in the vector (`rdn/LightTree.h:991–1002`; the TLAS builder and refitter use the same pattern).

An independent traversal of the uploaded fixture found:

| BLAS allocation | Nodes | Bytes |
| --- | ---: | ---: |
| Uploaded | 699,049 | 44,739,136 |
| Reachable from roots | 349,525 | 22,369,600 |
| Unreachable | 349,524 | 22,369,536 |

Building directly into reserved slots or compacting reachable sibling blocks would roughly halve this fixture's allocation and uploads, and reduce its address/page span. **It does not automatically halve L2 traffic:** unreachable nodes are not explicitly traversed. Apply the change to both initial builds and refits, preserve sibling order/trails, and invalidate learned node references when indices change. Audit results are in `tree-audit.log`.

## The proposed screen-space hash grid

Sharing one light selection per tile/normal/depth group can eliminate redundant descents. It should share the selected triangle/candidate, while each receiver still samples or evaluates the light point, evaluates its BSDF and geometry, and traces its own visibility. One shadow result or one radiance result per group would produce incorrect edges and shadows.

The central issue is the sampling distribution. Let a representative receiver generate light `l` with probability `q_rep(l)`. A consuming pixel must account for that actual proposal, not replace it with its own independently evaluated `q_pixel(l)`. For a uniformly sampled point on a triangle, the receiving pixel also needs its own area-to-solid-angle conversion. Emitter-hit MIS must use the proposal actually chosen for the previous vertex. Resampling a candidate pool by per-pixel importance requires the corresponding RIS/reservoir normalization rather than treating the winning candidate as an ordinary tree sample.

Normal/depth similarity alone does not prove compatible support. The current importance function can assign zero probability after receiver-horizon rejection; another receiver in the group may still receive that light. A representative proposal needs conservative support over the group or a positive-support fallback. Material IDs are not required for correctness if each pixel shades independently, but roughness and lobe differences can make a shared pool poor for variance.

One candidate is the most correlated setting: an entire patch may choose the same blocked or bright light in a frame. Start a candidate-cache prototype with a small pool, for example four candidates for a compatible 4×4 or 8×8 region, then measure quality and sharing rate. With one occupied surface group, this means 4× or 16× fewer tree descents respectively; normal/depth subdivisions reduce that saving. These are work-count ratios, not frame-time predictions.

Use screen tiles with bounded groupshared storage and checked keys before adding a global screen hash. Rebuild these groups each frame to avoid stale depth/normal associations. Keep the existing world-space learned grid for history and secondary receivers. Do not replace it with a camera-only grid. Any candidate pool must use explicit sampling and MIS normalization; it would be a different design from the current frozen-cut sampler.

## Recommended order

1. **Separate primary-prefetch and bounce-kernel diagnostics within the existing PT timer.** Attribute load stalls to tree nodes, learned cuts, geometry/BVH, material textures or spills. A long-scoreboard stall is a dependency symptom and can be reported at the consumer of an earlier load; it is not itself proof of L2 bandwidth saturation. See the [Nsight shader-profiler guidance](https://docs.nvidia.com/nsight-graphics/UserGuide/shader-profiler.html).
2. **Prototype small groups with correlated tree random numbers in the actual PT path.** Keep per-receiver traversals/PDFs and independent BSDF and light-point streams. This preserves each pixel's marginal proposal and avoids the representative-PDF problem. Vary group size and refresh pattern; compare equal-time variance and visible patch noise, including glossy and moving-camera views. The synthetic results make this the strongest near-term candidate.
3. **Separate frozen cuts from counters and remove unused uploaded nodes.** These are concrete layout/allocation improvements, but their PT-time benefit still needs measurement. Consider a dense alias table if scalar CDF dependency latency remains visible.
4. **Reduce expensive LOD/MIS traversal work.** One candidate is selecting and retaining a fine/coarse component per vertex, then conditioning both NEE and emitter MIS on that same choice. This can avoid eagerly evaluating the second subtree on every NEE sample, but changes variance and requires corresponding MIS/training tests. Simply dropping the mixture PDF term is incorrect.
5. **Then evaluate a small screen-tile candidate pool and compressed tree nodes.** Preserve per-pixel visibility and explicit proposal accounting. For secondary-hit sharing, group current world-space receivers or use the existing spatial grid; camera depth/normal bins describe the wrong surfaces.

Hardware counters and equal-time scene images are the remaining evidence needed to choose a production default. The experiments establish a substantial locality opportunity, not a whole-frame speedup or an image-quality result.
