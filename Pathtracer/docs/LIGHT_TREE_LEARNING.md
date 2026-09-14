# Learned light cuts

Enable **Integrator > Light sampling > Learn light clusters**. The renderer
learns visibility-weighted light cuts in path tracing and SHARC training.
When enabled, every receiver uses the learned cut as one of its light-sampling
techniques.

## Surface policy

`LTC_UseSurfaceLearning()` gates the feature globally. `LTC_TrainShare()`
chooses whether feedback uses the full or broad contribution according to
roughness and clearcoat; it does not remove glossy receivers from sampling.

NEE, emitter-hit MIS, and primary prefetch use the same learned-versus-ordinary
decision. The underlying BSDF, visibility rays, and path throughput are
unchanged.

## Spatial cells

Cells use absolute world position and quantized normal, so camera motion and
origin rebases preserve their keys. The requested width is quantized in powers
of two from a one metre minimum; default growth is 0.05, giving a 40 metre first
transition. Six normal-separated roots provide an immediate fallback after a
reset.

The table has 65,536 slots shared by all LODs. Each key probes two four-slot
buckets. Requests target the desired LOD directly, inherit from an available
coarser cell, and try a bounded fallback when both buckets are full. Idle nearby
history is retained, while stale detail can be reclaimed after its age limit.

LOD transitions blend fine and coarse proposals over the final quarter of the
logarithmic interval. Both cells receive every observation during the blend,
including zero rewards, and the same mixture PDF is used for sampling and MIS.
Published keys and cuts remain immutable for a rendering frame.

## Learning and storage

Cuts contain at most 32 entries. Initial Q values are calibrated to local
contribution scale, children recalibrate independently, and refinement can merge
low-value sibling groups before splitting. Feedback uses wave aggregation and
native 64-bit integer accumulation; five words represent the complete positive
FP32 range. Updates publish a frozen CDF after the UAV barrier.

The design follows Wang, Wu, Li and Chuang, *Learning to Cluster for Rendering
with Many Lights* (2021), [DOI](https://doi.org/10.1145/3478513.3480561). Wave
operations follow the [HLSL Shader Model 6.5 specification](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_ShaderModel6_5.html).

## Verification

Run `tests/run_light_tree_tests.ps1` for topology, bounds, sampling, feedback,
LOD transitions, allocation pressure, timer wrap, integer accumulation, and
record reuse. Use `-SurfaceTest`, `-ColdLodTest`, `-GridReviewTest`, and
`-Benchmark` for focused checks. Run
`tests/run_render_pipeline_tests.ps1` to verify pass selection and history
controls.

The feature changes the light proposal only. GPU timings and isolated integral
checks do not establish whole-scene image quality or convergence.
