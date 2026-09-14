Run `./tests/run_light_tree_tests.ps1` from a Visual Studio Developer PowerShell.
The standalone D3D12 runner compiles the production light-tree and update HLSL.

Coverage:

- Learned sampling for every receiver when enabled, including rough and clearcoat surfaces. Training feedback selects full or broad contributions by roughness and clearcoat. Sampling and emitter MIS share the same PDFs; a concentrated BSDF fixture checks the unit integral. Use `-SurfaceTest` for the focused checks.

- TLAS/BLAS trails at depths 0, 16, 17, 31 and 32, malformed depth-33 rejection, and independent topology/PDF walks through skewed and coincident scenes.
- Conservative cones, affine initial/refit bounds, topology-preserving origin shifts and stale-refit invalidation.
- Frozen normalized sample/MIS probabilities, known integrals, zero observations, variance reduction, refinement and reset.
- A 100 km camera jump, twice as many receivers as spatial slots (131,072 receivers with the current 65,536-slot table), and local-detail recovery for 512 new receivers after leaving a full table. Builds with at least 131,072 slots also check a fixed 65,536-receiver workload.
- Competing keys sharing a primary hash bucket, turning away/back while retaining exact cell indices and histories, frame-rate-independent retention and millisecond-counter wrap.
- Idle cells with partial batches becoming replaceable; actual feedback protecting active cells.
- Short camera movement with both fine buckets occupied by nearby idle history: obtain fine detail within 320 ms, preserve the inactivity grace period across timer wrap, and cancel an already queued replacement when feedback arrives later in the same frame. Uncontended turn-away history remains intact. Included in `-GridReviewTest`.
- Mature-view adaptation and parent refresh while fine cells remain in use.
- Exhausted 31/32-entry cuts with 4,096 lights, newly important emitters, inherited cuts after moving closer, and repeated visibility changes. Partitions and sample/MIS PDFs are checked throughout merging and splitting, followed by an independent known-integral check.
- Resuming refinement beyond the old permanent inactivity limit, including bounded moment history after lifetime visit counts approach uint overflow.
- Bidirectional LOD sweeps with deliberately different fine/coarse distributions: continuity, normalization, positive support and matching sampled/MIS PDFs.
- A mature fine cell with a missing next LOD and an unrelated shared root, tested at LODs 0, 2 and 6: prepare the next coarse cell from local learning before blending, preserve the useful emitter probability across the boundary in both directions, and update the future coarse cell from four observations/frame. The pre-fix snapshot fails by blending toward the shared root. Use `-ColdLodTest` for this focused regression.
- Every mixture observation reaching both fine and coarse partitions, checked against independent sums and counts, including zero rewards.
- A four-pixel cell near the 40 m LOD boundary continuing to update/refine its dominant coarse proposal, plus rebuilding missing intermediate parents while retaining the fine cell. Use `-LodTest` for this focused regression.
- Interleaved wave feedback, partial final waves, both NEE prefetch tokens and stale-record rejection.
- Dim-light scale calibration, including sparse eight-sample batches and inherited priors.
- Persistent intermediate-bucket collisions with free fine slots, fully blocked fine buckets falling back to a learning spatial parent, and immediate fine-detail recovery when pressure clears. Active colliding keys must remain intact. Use `-GridReviewTest` for these and the scale checks below.
- Calibrated bright priors inherited into dim children, including a camera move from a spatial parent trained with real feedback to a child receiving contributions twelve orders of magnitude smaller. Local scale and the newly visible emitter must recover without resetting the cache.
- Native integer feedback accumulation against independent CPU 320-bit sums: every FP32 exponent, subnormal inputs, five-word carry propagation, 2,073,603 contending observations, exact zero counts, float decoding and record reuse.

Learning storage is 297.03 MiB for 65,536 spatial cells plus six roots, shared across all hierarchical levels. Each cell has at most 32 cut entries. Frozen entries remain 64 bytes; exact accumulators use 80 bytes per entry. Trails use `LightTreeTrail.h`, uploaded as `R32G32_UINT` and read as `uint2`.

Use `-Benchmark` for coherent/scattered sample, sample+feedback and sample+PDF timings at 1920x1080 queries, with mature cuts in stable LOD and transition-band cases. Warmup trains at each measured camera distance so the blend parent is actually resident. A separate 65,536-receiver workload measures a saturated directory. Timings are GPU medians of five dispatches and exclude full-frame rendering. `-ShaderDirectory <path> -Benchmark` selects another sampler; add `-TestShader <matching test HLSL>` when comparing a snapshot with a different test-facing API.

Use `-HotCellBenchmark` for one-cell and one-cluster contention, from 65,536 through 8,294,400 queries. These are medians of three dispatches, consuming feedback after each dispatch; runs stop scaling a case once its median exceeds 100 ms. The scattered benchmark also measures both prepare dispatches with active and idle receivers. Performance comparisons should run sequentially without other GPU tests or renderers competing for the device.

Run `./tests/run_render_pipeline_tests.ps1` for pass selection and reconstruction-history controls. Further estimator, memory, timing and implementation details are in [LIGHT_TREE_LEARNING.md](../docs/LIGHT_TREE_LEARNING.md).
