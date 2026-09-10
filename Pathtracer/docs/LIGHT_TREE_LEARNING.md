# Light-tree sampling and learned cuts

The renderer now supports a spherical-Gaussian (SG) proposal and online learned light cuts. Both controls are in **Integrator > Light sampling**. SG applies to either integrator. Learned clustering applies to the regular path tracer, including its SHARC training paths, and can run with the radiance cache disabled. The legacy replay integrator retains deterministic light selection across replayed paths.

## Sources and scope

The attached paper is Wang, Wu, Li and Chuang, **Learning to Cluster for Rendering with Many Lights** (2021), [DOI](https://doi.org/10.1145/3478513.3480561). Its main contribution is learning and refining a cut on an existing light hierarchy using observed contributions, including visibility. SG lighting is a separate technique.

Inspected Tom Clabault's dev branch at `d1b3006b9aaa888e529aa571a8378806cb7dff8b`, particularly:

- [SG construction](https://github.com/TomClabault/HIPRT-Path-Tracer/blob/d1b3006b9aaa888e529aa571a8378806cb7dff8b/src/Renderer/LightTree/LightTreeSGBuilder.cpp): spatial moments, directional moments and fitted lobes over his ATS hierarchy.
- [SG sampling](https://github.com/TomClabault/HIPRT-Path-Tracer/blob/d1b3006b9aaa888e529aa571a8378806cb7dff8b/src/Device/includes/LightSampling/LightTree/LightTreeSGSampling.h): spatial/emission lobe products and diffuse/specular importance.
- [Learned-cut selection](https://github.com/TomClabault/HIPRT-Path-Tracer/blob/d1b3006b9aaa888e529aa571a8378806cb7dff8b/src/Device/includes/LightSampling/LightTree/LightTreeSGSamplingLearningToCluster.h): select a learned cluster, then descend its subtree, retaining the conditional probability.
- [Q updates](https://github.com/TomClabault/HIPRT-Path-Tracer/blob/d1b3006b9aaa888e529aa571a8378806cb7dff8b/src/Device/kernels/IlluminationAwareKDTree/LearningToCluster/LearningToClusterQUpdates.h) and [refinement](https://github.com/TomClabault/HIPRT-Path-Tracer/blob/d1b3006b9aaa888e529aa571a8378806cb7dff8b/src/Device/kernels/IlluminationAwareKDTree/LearningToCluster/LearningToClusterRefineLightcuts.h).

This is an original implementation for the renderer's two-level, four-way hierarchy. It does not import Tom's GPL source. The SG proposal is a simpler positive moment fit inspired by [Tokuyoshi et al., Hierarchical Light Sampling with Accurate Spherical Gaussian Lighting (2024)](https://doi.org/10.1145/3680528.3687647); it is not a reproduction of that paper's accurate anisotropic GGX integration or Tom's multiple spatial lobes.

## Sampling and learning

Receiver cells use quantized world position and normal, with exact key validation in a bounded hash table. The default spatial width is one metre and can be changed in the UI. Hash collisions fall back to the ordinary tree; an idle cell can be replaced after 256 frames. Requests are published only by the next prepare pass. Neither a cache miss nor concurrent training changes the current frame's sampling distribution.

Each cell starts with a power-prioritized cut of approximately four nodes, initialized from SG/cone importance. Its cut can cross TLAS leaves into the corresponding BLAS. A cut entry identifies the subtree and its 64-bit path prefix; every triangle belongs to exactly one entry. The maximum cut size is 32.

A cluster is selected with probability `0.95 * Q / sum(Q) + 0.05 / cutSize`, or uniformly if all Q values are zero. Sampling then descends the selected subtree. MIS evaluates that same product of cluster probability and conditional subtree probability. The distribution is frozen for the entire frame, including prefetch, SHARC training and emitter-hit PDF evaluation. The emitter-hit query retains the actual previous receiver and normal; recovering them from ray distance or packed normals could change the cell.

Feedback uses the local, visible BSDF-weighted contribution divided by the actual light PDF, before MIS and path-throughput weighting. Multiplying by the selected cluster probability produces a conditional cluster observation. Back-facing/invalid geometry, occluded lights and zero BSDF contributions submit zero observations. The update converts accumulated conditional observations back to a full-batch importance estimate using the frozen cluster probability and total batch size. Unvisited clusters therefore receive their required implicit zero observations.

The schedule is the paper's `alpha(t) = 1 / (4 * t^(6/7))`. Batches contain at least `4 * max(ceil(cutSize / initialCutSize), 2)` samples; undersized batches span frames. Conditional means and variances are accumulated with a merged-moment update. Refinement follows the variance, relative variance, visit-count and cut-growth factors of Eq. 7, with at most one stochastic split per cell per batch. Children use Eq. 8's prior/parent initialization, generalized to four children. Refinement stops at capacity or after the paper's inactivity criterion with Gamma=128.

Adaptations relative to the paper: a fixed spatial/normal hash replaces the preprocessing 5D BVH, cuts are four-way and capped at 32, and the maximum of one split per batch bounds GPU work. The actual conditional PDF is retained in training rather than substituting the paper's practical geometric-mean approximation. These choices are not a claim to reproduce the paper's equal-time image results or its full convergence theorem under finite-precision, finite-capacity operation.

## SG representation and tree corrections

Nodes carry power-weighted spatial means/variances and a fitted emission SG. CPU covariance accumulation is centered and uses doubles. Uniform-triangle covariance and the Lambertian directional first moment are analytic. The proposal integrates a product of spatial, reversed-emission and positive receiver-cosine SGs. A 5% conservative cone contribution preserves support when the fit is poor. The true triangle emission, visibility and material remain in the radiance estimator.

Initial TLAS bounds now transform object-space BLAS roots into world space. The builder and refitter share deterministic instance ordering, so BLAS identities remain consistent. Refits initialize cones from the first emitter instead of unioning everything into an initially full sphere. Antiparallel cone merging preserves asymmetric cone widths. Affine transforms update spatial covariance; nonuniform transforms use isotropic emission proposals and conservative full orientation cones. Receiver normals use the appropriate pullback for BLAS queries. Near-field distance regularization uses squared radius rather than mixing squared distance and linear radius.

Emissive membership or power changes rebuild both tree levels, SG moments and trail tables, and discard obsolete asynchronous TLAS results. Published TLAS changes, geometry/material changes, learning/SG controls, cell size and shader reload reset learned references. Ordinary frame updates do not reset learning.

## Cost and verification

Nodes increase from 64 to 96 bytes. The learning table adds **34 MiB**, independent of resolution, to the existing persistent allocation. It requires no extra root-signature DWORDs. Slot 24 carries cell width under PT and retains its existing SPMIS meaning under legacy ReSTIR. Reset bits for SHARC and learned lighting are independent.

Run `tests/run_light_tree_tests.ps1` and `tests/run_render_pipeline_tests.ps1` from a Visual Studio Developer PowerShell. The GPU suite covers existing depth-32 paths and malformed depth-33 trees, CPU/GPU layouts, affine moments, cone support, scale consistency, stale-refit discard, frozen-frame PDFs, zero observations, normalized full-support cuts, TLAS-to-BLAS refinement, reset, and a known unit-integral fixture.

The light-tree GPU suite, render-pipeline tests, SHARC/material/reuse regressions, and the full RelWithDebInfo application build passed on the development machine.

For the deterministic 128-emitter fixture with one visible emitter, learned cuts grew from 4 to 11 entries. Across three configurations, estimated integrals were 0.9968-1.0046. Second moments decreased from 128.4-135.7 to 2.27-2.57 (the true mean is one). This validates a controlled sampling improvement; it is not a full-scene visual comparison or an equal-time performance result. The simpler SG fit can help or hurt on individual distributions; use its toggle for scene comparisons.

Finite floating-point feedback sums and importance updates saturate at their learning limits, and nonfinite feedback is treated as zero. This protects metadata and does not clamp the rendered contribution. The hash/cut limits and quantized receiver sharing limit adaptation quality, especially across rapidly varying visibility or view-dependent materials. Continuous geometry edits reset learning; moving-scene temporal reuse is not implemented.
