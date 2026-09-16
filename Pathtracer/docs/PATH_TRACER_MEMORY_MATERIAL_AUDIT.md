**Path tracer memory and material audit**

The highest-value opportunities are texture traffic, material execution structure, state crossing ray/reorder boundaries, and unnecessary full-screen work. The material system also has correctness defects that should be fixed before treating its current performance as the baseline.

This audit covers the regular PT, SHARC training and guiding, learned lighting, Lite reuse, legacy ReSTIR, camera/visibility shaders, atmosphere/cloud passes, reconstruction inputs, exposure, postprocessing, resource allocation and material importing.

Evidence is distinguished below:

- **Code:** directly established by current source or compiled DXIL.
- **GPU diagnostic:** freshly executed production material functions with explicitly described fixtures.
- **Candidate:** an optimization that needs a representative whole-frame A/B profile.
- **Reference:** a source or API link used to locate the relevant implementation.

**1. Priorities**

| Priority | Change | Main benefit | Evidence / qualification |
|---|---|---|---|
| First | Correct divergent material descriptor indexing | Correct texture selection; trustworthy profiling baseline | Current source and newly compiled DXIL mark varying texture indices uniform. |
| First | Correct delta events, NEE eligibility and material/import semantics | Stable, physically meaningful materials | Concrete code defects; separate from cache tuning. |
| High | Preserve/cook BC textures and select mips from ray footprints | Smaller texture working set; less L2/VRAM traffic and aliasing | Imports currently expand textures to RGBA8; primary shading uses mip 0. Actual timing gain depends on scene. |
| High | Specialize common material classes; prepare only active lobes; fuse sample/evaluation work | Less ALU, instruction footprint and register pressure | Generic material paths repeatedly construct the same quantities. Driver elimination must be measured. |
| High | Profile and reduce current SER live state; move coherent hit preparation to the useful side of reordering | Less spill/reload traffic; more coherent texture/material accesses | Existing packing is already useful; fresh backend measurements are required. |
| High | Gate reflection-guide rays and debug/reference output work by actual consumers | Fewer rays and full-screen reads/writes | Camera probe is unconditional after valid surface resolution; debug slices default on. |
| Medium | Allocate resources per enabled mode; split scratch by lifetime/format | Substantial VRAM reduction; some bandwidth savings | Allocation is much larger than the active PT state. Allocation size alone is not L2 bandwidth. |
| Medium | Separate hot light/cache query data from training data, retaining compact spatial capacity | Better cache locality without reviving allocation starvation | Keep the existing 65,536-slot learned grid and movement recovery while testing layout changes. |
| Later | Selective queues for expensive SSS/media; resource-specific barriers | Lower common-path pressure and better scheduling | Larger changes; queue traffic and synchronization can outweigh gains. |

**2. Memory budget: the renderer allocates much more than the material table**

The authoring material record is **48 bytes**. Vertex records are **20 bytes**, instance records **224 bytes**, light triangles **80 bytes**, and current light-tree nodes **64 bytes**. Accessing one field does not necessarily fetch every byte of its record. Conversely, a 48-byte material stride does not mean each material occupies exactly one cache line. See [GPU records](../shaders/Data_v8.hlsli).

The following are logical allocation sizes, before resource alignment, at the allocation dimensions. Listed resolutions are tile aligned. They are **not bytes transferred every frame**.

| Screen resource | Bytes/pixel | 1920×1080, MiB | 3840×2160, MiB |
|---|---:|---:|---:|
| Six RGBA8 output slices | 24 | 47.5 | 189.8 |
| Deferred cloud-query records | 64 | 126.6 | 506.3 |
| FP32 reference accumulation | 16 | 31.6 | 126.6 |
| Fifteen RGBA32F scratch slices | 240 | 474.6 | 1,898.4 |
| Two reservoir buffers | 160 | 316.4 | 1,265.6 |
| Two surface-sample buffers | 72 | 142.4 | 569.5 |
| Path/reuse scratch | 168 | 332.2 | 1,328.9 |
| SPMIS arrays/search records | 72 | 142.4 | 569.5 |
| Legacy survivor queue | 4 | 7.9 | 31.6 |
| Four old streaming stacks | 32 | 63.3 | 253.1 |
| **Listed total** | **852** | **1,684.9** | **6,739.5** |

Sources: [screen allocations](../rdn/Renderer_Pipeline.cpp), [record sizes](../rdn/Renderer.h), [path scratch allocation](../rdn/Renderer.h), [streaming stacks](../rdn/Renderer_Pipeline.cpp). SPMIS has a small constant header and a minimum sky-bake size; other small headers are omitted.

Additionally, the shared cache allocation is **502.77 MiB**:

- SHARC: 1,048,576 × (160-byte entry + 4-byte state) = 164 MiB.
- Guiding: 262,144 × (160 + 4) = 41 MiB.
- SHARC dirty mask: 0.125 MiB.
- Lite pair tables: approximately 0.616 MiB.
- Learned lighting: **65,536 spatial slots plus six roots, 297.03 MiB**.

See [cache layout](../shaders/SharcLayout.h) and [learning layout](../shaders/LightTreeLearningLayout.h). The learned grid remains hierarchical/adaptive; levels share the table. Its large byte count comes from per-cell cuts and training accumulators, not millions of spatial slots.

DLSS resources add **67 bytes per input-resolution pixel plus 8 bytes per output-resolution pixel**, excluding SDK internals and optional neural-rendering resources. Scene textures, geometry, acceleration structures, cloud volumes, sky resources and driver RT stack memory are also excluded from the table. See [DLSS allocations](../rdn/PostProcess/DLSSManager.cpp).

Concrete allocation changes:

1. PT without Lite does not need two full legacy reservoir buffers. Lite needs its own smaller layout, not necessarily the full legacy capacity.
2. PT needs the sky-bake prefix of SPMIS, not its full 72-byte/pixel layout.
3. Create legacy queue/streaming resources only when their pipeline is used.
4. Allocate cloud query storage when clouds are active.
5. Split the 15-layer scratch array into resources with declared lifetimes and formats. Keep world positions and large-range depth in FP32; pack normals, bounded transmittance and masks where their consumers permit it.
6. Preserve mode switching through allocation/rebinding and history invalidation. Existing descriptors and aliases must be updated together.

The stale comment in [Path_State](../shaders/Path_State_v8.hlsli) still says 88 bytes/pixel; the actual CPU allocation is 168. Use the allocation and active access ranges as the contract.

**3. Texture traffic is a larger target than material-record compression**

**Code:** both OBJ and glTF import paths build RGBA8 mip chains. Compressed DDS images are decompressed before this upload path. The GPU uploader retains that uncompressed format. See [OBJ processing](../src/Util/ObjLoader.h), [glTF processing](../src/Util/ObjLoader.h) and [upload](../rdn/Scene/AssetLoader.cpp).

A 4096² RGBA8 texture with a full mip chain is approximately **85.3 MiB**. BC7 is approximately **21.3 MiB**: one quarter of the encoded storage. This is not a guaranteed fourfold bandwidth or frame-time improvement; cache access patterns and filtering matter. BC formats use blocks rather than independent compressed texels. [Microsoft block-compression reference](https://learn.microsoft.com/en-us/windows/win32/direct3d10/d3d10-graphics-programming-guide-resources-block-compression).

Recommended texture pipeline:

- Preserve suitable precompressed DDS assets; cook color textures to BC7 sRGB, normal XY to BC5 with reconstructed Z, and masks to an appropriate BC format.
- Keep a high-quality uncompressed/16-bit option for assets where compression damages smoothness, normals, displacement or fine opacity.
- Generate coverage-preserving alpha mips; coordinate those with OMM/any-hit behavior.
- Filter normals and roughness together so lost normal-map variance becomes appropriate specular roughness.
- Deduplicate cooked textures by source, color-space and processing settings; avoid duplicating a texture solely because two materials have different factors.

**Code:** primary material fetches use level 0, while secondary material and normal-map fetches use bounce depth as mip level. Bounce number is not a footprint: distant primary geometry overfetches, while a nearby secondary reflection may be overblurred. See [primary fetch](../shaders/Pass_camera_v8.hlsl) and [secondary fetch](../shaders/Pass_pt_v8.hlsl).

Use ray cones or differentials, propagated through reflection/refraction and converted through the UV Jacobian, to choose mips. The additional cone state should be a few scalar values, not a large differential struct live through every trace. Preserve point-filter mode while still choosing a suitable mip. Alpha visibility needs a consistent footprint policy rather than a disconnected change to its mip-0 test.

**4. Slim the executed material, not its feature list**

The current record is compact; the expanded execution is the problem. [Sampling probabilities](../shaders/BXDF_v8.hlsli), [sampling](../shaders/BXDF_v8.hlsli), [combined evaluation](../shaders/BXDF_v8.hlsli) and [GGX evaluation](../shaders/Material_GGX_v8.hlsli) repeat material decoding, frame construction, normalization, Fresnel, alpha mapping, masking and LUT work. Some repetition can be eliminated by DXC; it is not valid to count every source-level accessor as a separate memory load.

A practical replacement has three distinct representations:

| Representation | Contents | Lifetime |
|---|---|---|
| Packed material definition | Factors, texture handles, feature/class flags, extension reference | Persistent GPU data |
| Prepared shading point | Textured parameters and frame; only enabled lobe parameters; outgoing-direction terms reused by NEE and continuation | One shading vertex |
| Compact continuation | Throughput, ray, exact required previous-position/PDF state, event flags and medium state | Across ray tracing/reordering |

Implementation direction:

1. Use a small set of material classes: opaque isotropic, opaque anisotropic, solid dielectric, thin dielectric, layered surface, and SSS boundary. Retain a general fallback. Classify by features, not thousands of individual material IDs.
2. Specialize cheap common cases at compile time or through compact class kernels. A constant-false runtime feature still contributes to the generic shader's compiled code and may affect its resource requirements.
3. Prepare invariant fields once: active-lobe mask, textured roughness/metalness, alpha axes, tangent frame, IOR/F0 and needed outgoing LUT terms. Do this after the relevant reorder; release it before the next trace.
4. Provide one sample-and-evaluate entry that reuses its sampled half-vector and intermediate geometry. It must return the full mixture PDF, valid/null/delta flags and correctly layered contribution. Keep a separate arbitrary-direction evaluator for NEE/reconnection, sharing the same closure formulas.
5. Retain the already-fused full/broad evaluation. That optimization exists; duplicating it is not new work.
6. Keep FP32 for PDFs, density Jacobians, narrow-lobe geometry, accumulated radiance and large coordinates. Bounded colors and scalar parameters can remain packed/half where validated.
7. Experiment with a **32-byte common record plus cold extensions** for texture transforms, coat/sheens/SSS extras. This is a design target, not a proven optimal size. First profile which fields are actually fetched; an extra dependent extension load can be worse than the existing 48-byte record.

Do not keep all decoded lobes alive across a trace merely to avoid a few cheap loads. Likewise, do not pad all materials to 64 bytes just to obtain alignment without measuring the additional traffic.

The existing 16×16 material LUT is generated from separate CPU implementations, with nondeterministic random seeds. It is tiny enough that its capacity is unlikely to explain a giant L2 bottleneck. However, it adds dependencies and repeated sampling. Generate deterministic reference LUTs from matched math, validate their interpolation, and compare compact analytic fits. The CPU samples at x/(N−1), but lookup passes that parameter directly as normalized texture coordinates; endpoint-grid lookup should use (u·(N−1)+0.5)/N with clamp addressing. See [generation](../rdn/Renderer_Pipeline.cpp) and [lookup](../shaders/Material_Common_v8.hlsli).

**5. Register pressure and live state**

The current PT already packs normals and reconstructs IOR/absorption after its reorder, uses a four-byte payload, an eight-byte attribute record, and recursion depth one. These are good foundations. See [reorder boundary](../shaders/Pass_pt_v8.hlsl) and [DXR configuration](../rdn/Renderer_Pipeline.cpp).

There are still specific candidates:

- **Reorder placement:** expensive hit/material/normal-map preparation currently precedes the next reorder. Its key uses instance low bits. Test cheap hit classification followed by material-class/texture-coherent reordering before that preparation. Compare against the current single reorder; an additional reorder can cost more than it saves.
- **Live radiance and reuse state:** trace the lifetimes of total radiance, throughput, Lite suffix state, previous MIS state, training receiver state and hit data. Use feature variants so disabled Lite/guiding/SSS genuinely disappear from the compiled hot path.
- **Guiding:** an eight-cone GuideSet is 80 logical bytes before temporary cone-building state. The current code already builds it after NEE. Reduce the peak construction state and ensure the cone set dies after sampling/PDF calculation, rather than spilling it through the next trace. See [GuideSet](../shaders/SharcGuide_v8.hlsli).
- **Training spills:** two RGB radiance/weight pairs deliberately use 48 bytes of scratch per training pixel. Test a compact training-lane index and SoA/AoSoA layout against the current pixelIndex×48 addressing, especially with sparse training. This trades register state for explicit traffic; keep both sides in the measurement. See [training state](../shaders/SharcPath_v8.hlsli).
- **SSS:** the long, divergent random walk is a good candidate for an exceptional-material queue or separate specialized shader, avoiding its peak requirements in common opaque work.
- **Light traversal arrays:** four child weights and topology records are currently retained together. Compare keeping only hot traversal data, or reloading the winning child's topology, against the present layout. Fewer registers may introduce another dependent load; neither is automatically faster.

The **MAX_REGS=96 define does not cap registers**: it is passed to DXC but is not consumed by shader code. [Compiler arguments](../rdn/DXRHelper.h). HLSL half declarations, DXIL SSA IDs and alloca counts are also not physical register counts.

Existing Nsight evidence reports PT reorder call sites with **275 and 221 bytes of live state per thread**. These values require fresh backend captures before they are used as current cost or occupancy data.

For the next A/B, capture actual driver register usage, RT live-state bytes per call site, local spill traffic, occupancy, active lanes, L2 sectors/bytes, texture traffic, DRAM bytes and long-scoreboard stalls. A high L2 throughput value can mean cache hits serving excessive traffic; a miss bottleneck needs different treatment. Inspect live state at ray/reorder boundaries rather than inferring physical registers from HLSL variables.

Queue based execution can reduce divergence and peak state but adds memory transactions and dispatches at every stage. Start with common material specialization and selective expensive paths.

**6. Material correctness findings**

**6.1 Divergent descriptor indexing — fix before performance tuning**

Material-dependent albedo, RMA, normal and alpha texture indices use ResourceDescriptorHeap[index] without NonUniformResourceIndex. The newly compiled current PT DXIL contains resource-heap CreateHandleFromHeap calls with nonUniformIndex=false. Instance-based SER does not guarantee uniform material or texture identity across a wave.

Fix the varying resource indices in [material/alpha fetches](../shaders/Inline_RT_v8.hlsli) and [any-hit](../shaders/AnyHit.hlsl). The point/linear sampler selection is a uniform frame setting and does not need the same annotation. Omitting the intrinsic for a varying heap index permits undefined results under the API. [DirectX dynamic-resource specification](https://microsoft.github.io/DirectX-Specs/d3d/HLSL_SM_6_6_DynamicResources.html).

This establishes an API defect, not proof that it caused the previously reported learned-cell problem.

**6.2 Smooth sampling and continuous evaluation disagree**

[GGX sampling](../shaders/Material_GGX_v8.hlsli) and [coat sampling](../shaders/Material_Coat_v8.hlsli) force the half-vector to the normal below roughness 0.06. Their evaluators still report a finite-width lobe and continuous density. PT carries that density into emitter/sun MIS.

Either retain finite-width VNDF sampling for nonzero roughness, or implement real delta events with discrete event probabilities and matching throughput/MIS behavior. An ideal delta reflection hitting an ordinary area emitter must not be discounted by a competing continuous light density. This is also an opportunity for a very cheap, correct mirror/glass path.

**6.3 MIS includes techniques that were disabled**

[PT disables NEE](../shaders/Pass_pt_v8.hlsl) inside a medium and for fully transmissive materials, but [sun hits](../shaders/Pass_pt_v8.hlsl) and [emitter hits](../shaders/Pass_pt_v8.hlsl) still use competing light PDFs. Where those PDFs are nonzero, radiance is discounted without a complementary light-sampling contribution.

Carry a compact record of which techniques were eligible at the preceding vertex, alongside delta/event flags. The diffuse continuation limit also occurs after NEE: explicitly define its truncated-path policy and avoid weighting NEE against a continuation that is never drawn.

Do not mechanically replace every MIS density with the guiding density. The existing guiding code intentionally uses complementary weights based on the unguided BSDF while dividing each estimator by its actual proposal. Complementary weights can be valid; tests must check the whole estimator.

**6.4 Layering is not reciprocal**

Clearcoat and base GGX attenuation scale the outgoing Fresnel by (1−0.7·roughness)² but do not apply the same construction to the incoming side. Sheen attenuation depends only on the outgoing directional albedo. GGX/coat energy compensation also uses an outgoing-only multiplier. See [coat](../shaders/Material_Coat_v8.hlsli), [GGX](../shaders/Material_GGX_v8.hlsli) and [sheen](../shaders/Material_Sheen_v8.hlsli).

Fresh GPU diagnostic: view cosine 0.15, light cosine 0.85, coat weight 0.75, coat roughness 0.2 and IOR 1.5 produce coat substrate transmission **0.719247** versus **0.636072** after swapping directions: a ratio of **1.13076**. This attenuation result does not depend on the LUT.

The diagnostic also evaluates full production BSDFs with a deterministic analytic LUT fixture:

| Fixture, base roughness 0.4 | f(light, view) / f(view, light) |
|---|---:|
| Opaque dielectric | 1.37714 |
| Metal | 1.18523 |
| Coated dielectric | 1.55359 |
| Coated dielectric with sheen | 1.60786 |
| Anisotropic dielectric | 1.39164 |

These ratios are not measured appearance errors in the city scene and are not a production-LUT white-furnace test. They demonstrate how current formulas break reciprocity in controlled reflection cases.

Use reciprocal layer coupling and a validated multiscattering model. Directional renormalization can recover energy while violating reciprocity; it should be a documented approximation, not a claim of a physically complete layered model.

The combined evaluator also uses sampling-probability thresholds to decide whether to evaluate physical lobes and their attenuation. Separate an explicit active-closure mask from the proposal probabilities: changing a sampling heuristic should change variance, not the material's scattering function. Keep intentional cache-owned lobe removal explicit in that mask.

**6.5 Geometry and anisotropy need a real surface frame**

[The anisotropic basis](../shaders/Material_Common_v8.hlsli) is constructed from world normal and an arbitrary axis, with a switch near |Nz|=0.999. It ignores the UV tangent. Brushed direction therefore does not reliably follow the asset, and the frame can jump.

Signed anisotropy is decoded in [−1,1], but alpha construction uses abs(anisotropy). GPU checks give identical axes for +0.8 and −0.8: **(0.302372, 0.084664)** for input alpha 0.16. Preserve the intended sign by swapping axes/rotating the direction, or explicitly migrate to a nonnegative-strength-plus-angle convention.

PT also supplies the shading normal as both shading and geometric normal in [BSDF sampling/evaluation](../shaders/Pass_pt_v8.hlsl), defeating geometric-side support checks. Carry the true geometric normal, preserve UV tangent handedness under mirrored/nonuniform transforms, and define a consistent shading-normal correction. Separate denoiser guide stabilization from the actual transport normal.

**6.6 Medium transport is too approximate for production glass**

[Next-hit medium setup](../shaders/Pass_pt_v8.hlsl) infers air/material IOR solely from the next hit's facing and material. This is not nested-medium tracking. Absorption is attached to destination material/scatter state, while emission is handled before that setup.

Maintain the current medium explicitly, update it only on transmission, and apply segment attenuation before shading any hit, including emitters. Add a small nesting stack or a well-defined priority scheme. Verify refractive radiance scaling and roulette etaScale with glass slabs and nested interfaces; the presence of eta² in the BTDF Jacobian alone does not establish correct transport scaling.

Thin glass currently approximates a single interface. For a nonabsorbing thin sheet with two interfaces, normal-incidence IOR 1.5 gives total reflectance **0.076923**, compared with single-interface **0.04**. Use matched thin-sheet reflection/transmission and shadow transmittance. Exact dielectric Fresnel should replace Schlick-plus-TIR where critical-angle behavior matters.

**6.7 SSS exists, but its boundary model needs replacement**

The current [SSS implementation](../shaders/Material_SSS_v8.hlsli) is a real homogeneous random walk. Its limitations are nevertheless substantial:

- It traces the entire scene without carrying the originating medium/object identity; an internal intersecting object can be treated as the SSS boundary.
- The exit normal uses objectToWorld rather than an inverse-transpose normal transform.
- The scalar mean free path cannot express RGB scattering distances.
- Entry uses a cosine approximation and exit is intended to become white diffuse rather than a coupled dielectric boundary.
- The PT exit retains the original material ID, so coat/sheen fields can remain active despite the white-diffuse intent.
- A path-wide entered-SSS bit prevents later independent SSS entries; the fixed 64-step cap discards surviving walks.

Use explicit medium boundaries, per-channel optical coefficients or calibrated radius/color conversion, proper boundary Fresnel/refraction, and an explicit exit closure. Keep bounded fast modes as deliberate approximations, with a reference random-walk mode for validation. The material ID should not implicitly describe both the exterior BSDF and every interior state.

**6.8 Importing already changes material meaning**

This is as important as the BSDF math:

- Albedo textures **replace** base-color factors; RMA textures **replace** roughness/metalness factors. The glTF import does not bake those products when textures exist. Shared images used with different factors therefore shade incorrectly. See [texture evaluation](../shaders/Inline_RT_v8.hlsli) and [glTF factors](../src/Util/ObjLoader.h). glTF defines factor/texture multiplication. [glTF material specification](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#materials).
- [Sheen import](../src/Util/ObjLoader.h) maps sheenRoughnessFactor into the sheen **weight** while the shader fixes sheen roughness at 0.2. Preserve color, weight and roughness separately. [KHR_materials_sheen](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_sheen).
- [Anisotropy import](../src/Util/ObjLoader.h) stores a radians value in the field later decoded as a normalized angle multiplied by π. Establish one explicit angle unit and convert during import. [KHR_materials_anisotropy](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_anisotropy).
- [Volume import](../src/Util/ObjLoader.h) deliberately turns absorption volume parameters into scattering/SSS and substitutes a unit mean free path for unspecified attenuation distance. Separate absorption from scattering; clear colored glass should not become a scattering solid. [KHR_materials_volume](https://github.com/KhronosGroup/glTF/tree/main/extensions/2.0/Khronos/KHR_materials_volume).
- Coverage opacity, refractive transmission and SSS share overloaded controls. They need independent parameters, even if packed tightly.
- Emissive textures are averaged at import, and emitter hits terminate transport. For emissive painted surfaces, retain textured emission, add emitted radiance, then continue the underlying BSDF unless the material is explicitly a pure emitter. Light sampling must use a matching textured emission distribution or a valid conservative proposal.
- processGltfRMA indexes AO at the MR image dimensions without resampling/checking differing image sizes; its cache key also omits constants used when an input is absent. Fix texture preprocessing independently of shading.

The last items are code findings in [RMA preprocessing](../src/Util/ObjLoader.h) and [emission import](../src/Util/ObjLoader.h); no asset-specific visual magnitude was measured here.

**7. A practical material target**

Use explicit mixing and layering semantics with a specialized GPU implementation rather than an ever-larger universal material struct. Keep opacity separate from transmission and track energy preservation and reciprocity for every approximation.

The target retains current features and adds controlled extensions:

| Current capability | Proposed implementation |
|---|---|
| Diffuse | Lambertian fast case plus a validated rough-diffuse closure |
| Dielectric/metal GGX | Matched VNDF sampling/evaluation, correct Fresnel, reciprocal energy compensation; optional RGB conductor optical parameters |
| Anisotropy/rotation | UV/object tangent frame, defined angle convention, consistent signed behavior |
| Clearcoat | Its own roughness, IOR and optional normal/tint; energy-consistent coupling to the substrate |
| Sheen | Independent color/weight/roughness and a reciprocal substrate attenuation model |
| Solid transmission | Rough and delta dielectric transport, explicit nested media, Beer attenuation |
| Thin transmission | Separate thin-sheet closure, including diffuse translucency where needed |
| SSS | Medium-aware random walk and calibrated RGB scattering distance; separate reference/fast policies |
| Alpha, normal maps and textures | Independent coverage, correct factor multiplication, footprint filtering and normal variance handling |
| Emission | Textured additive emission plus optional underlying scattering |
| Additional appearance | Cold optional records for thin film and other specialized effects |

This table is an implementation roadmap. Keep a versioned compatibility adapter for existing scenes because correcting semantics can change appearance.

A credible RGB approximation is achievable without requiring spectral rendering for every path. Spectral dispersion, polarization, dedicated hair/fiber scattering, displacement and full production volume transport are separate extensions. Surface BSDF quality alone also cannot make SHARC's view-independent broad-lobe approximation or denoising into a reference renderer; validation needs an uncached, unreused scene-linear path.

**8. Pass-by-pass opportunities**

Pass activation follows [the actual pass list](../rdn/Renderer.cpp) and [feature classification](../rdn/Raytracing/PassSystem.cpp). The PT timer includes NEE prefetch as well as the bounce dispatch. The list below covers active and selectable rendering paths.

| Pass / subsystem | Current work and recommended next step |
|---|---|
| Camera | Writes resolved primary material/surface data once, which is useful. Its additional reflection probe runs for every valid surface, including rough ones. Gate by actual guide/reuse consumers; test a separate guide pass so RayQuery state does not inflate all primary shading. Preserve stable guide semantics. [Source](../shaders/Pass_camera_v8.hlsl) |
| Any-hit / inline visibility | Existing opaque splitting and OMM are valuable. Fix divergent descriptors; keep alpha-only fetches small; consider a separate compact UV/opacity metadata stream. Preserve closest-hit versus visibility semantics. The 128-candidate limit needs stress tests on dense cutouts/glass. [Source](../shaders/Inline_RT_v8.hlsli) |
| PT primary NEE prefetch | Already separates primary tree descent from the material megakernel. Keep it until A/B evidence favors fusion. Experiment with tile scheduling, shared upper-tree data and compact sample tokens; preserve per-pixel proposal/PDF. [Source](../shaders/Pass_pt_nee_v8.hlsl) |
| PT bounce | Main material specialization, texture-footprint and SER/live-state work described above. Avoid adding full-screen queues for every bounce without evidence. [Source](../shaders/Pass_pt_v8.hlsl) |
| Learned-light resolve/update | Frozen sampling cuts and mutable training statistics have different access patterns. Test a compact sampling-only layout, contiguous CDF data and separation of mutable counters. Keep allocation/publication/reset ordering and parent/fine fallback intact. [Source](../shaders/Pass_light_learning_v8.hlsl) |
| Light-tree traversal/PDF | Upper nodes recur; lower traversal diverges. Profile hot topology/bounds separately from emission/statistics. Compare breadth-first upper levels, packed child blocks and winning-child reload against register arrays. Quantize bounds only conservatively and keep sample/PDF reconstruction identical. [Source](../shaders/LightTree_v8.hlsli) |
| SHARC prepare | Visits cache and guide state each frame. Replace repeated per-entry current/previous matrix comparisons with per-instance motion generations. Consider budgeted maintenance only with equivalent expiry/motion correctness. [Source](../shaders/Pass_sharc_prepare_v8.hlsl) |
| SHARC update | Sparse PT specialization already exists. Profile explicit 48-byte training spills, sample-density locality and exceptional materials. Keep sparse training from serializing atomic contention on a shared receiver. [Source](../shaders/SharcPath_v8.hlsli) |
| SHARC resolve | Dirty-mask resolve already avoids touching every payload. Next compare hot query mean/confidence against separate accumulation storage, rather than reintroducing a full sweep. [Source](../shaders/Pass_sharc_resolve_v8.hlsl) |
| Guiding | Eight-cone construction uses temporary arrays and multiple passes. Reduce construction peak state; consider progressive rejection before full slot loads. Do not independently reconstruct different cone sets for sampling and PDF. [Source](../shaders/SharcGuide_v8.hlsli) |
| Lite shift | Partner lookups and visibility dominate. Keep rejection metadata compact; load full samples only after cheap gates. Test compact dispatch only when active coverage is sparse. [Source](../shaders/Pass_lite_shift_v8.hlsl) |
| Lite merge | Already loads partner W/meta compactly and handles three partners. Avoid loading every complete reservoir before selection; evaluate whether its fixed arrays scalarize. Preserve pairwise MIS and winner visibility. [Source](../shaders/Pass_lite_merge_v8.hlsl) |
| Legacy raygen | Inactive in regular PT; already uses a compacted survivor dispatch. It shares the material defects. Keep lobe-indexed replay, null samples and selection probabilities consistent when replacing closures. [Source](../shaders/Pass_raygen_v8.hlsl) |
| Legacy temporal GI | Now prepares shift jobs without tracing/replay inside this dispatch. A compute conversion is a candidate; compare launch/occupancy benefits, not source size. [Source](../shaders/Pass_temp_gi_v8.hlsl) |
| Legacy temporal shift | Unified shift kernel already defers merge weights. Preserve that reduced live state; improve material-class coherence and post-trace load lifetimes in replay. [Source](../shaders/Pass_shift_v8.hlsl) |
| Legacy temporal merge | Pure weighting/merge plus deferred environment work. Read hot weights/status first and retain full payload only for the selected result. [Source](../shaders/Pass_temp_merge_v8.hlsl) |
| SPMIS reset | Full-resolution grid/search-key clears. A touched-cell list or epochs may save traffic, but require reliable miss sentinels and bounded overflow handling. Existing sparse payload initialization must remain correct. [Source](../shaders/Pass_spmis_reset_v8.hlsl) |
| SPMIS count | Several same-cell atomic additions per pixel. Wave-match aggregation can combine count/confidence and reserve contiguous per-class indices once per matching group. Preserve each member's rank and M accounting. [Source](../shaders/Pass_spmis_count_v8.hlsl) |
| SPMIS offsets | Already skips empty cells. Aggregate occupied counts per wave/group or use a compact occupied list before reserving from the global counter. Do not propose adding millions of zero atomics back. [Source](../shaders/Pass_spmis_offsets_v8.hlsl) |
| SPMIS sort | Already precomputes RIS target and builds a 32-byte search record. Preserve that good locality; review whether all consumers need the full normal/position precision and duplicate arrays. [Source](../shaders/Pass_spmis_sort_v8.hlsl) |
| SPMIS select | Existing batched candidate fetches improve latency hiding. Tune batch size against registers; keep candidate metadata compact and delay reservoir payload reads. [Source](../shaders/Pass_spmis_select_v8.hlsl) |
| SPMIS passthrough | Simple routing/copy path. Candidate for compute or fusion with the adjacent routing stage, respecting output dependencies. [Source](../shaders/Pass_spmis_passthrough_v8.hlsl) |
| SPMIS spatial shift | Same shift shader, depth slices represent jobs. A compact list could remove inactive slices; test queue cost and keep receiver/candidate mapping exact. [Source](../shaders/Pass_shift_v8.hlsl) |
| SPMIS merge | Reads raw shifted contributions and applies final weights; no reason to move these arrays back across ray tracing. Load winner payload late. [Source](../shaders/Pass_spmis_merge_v8.hlsl) |
| Duplication map | Already caches a 32×32 tile in groupshared memory for a 17×17 neighborhood. The remaining target is reduction work/shared-memory access, not repeated global reservoir loads. [Source](../shaders/Pass_dup_gi_v8.hlsl) |
| Atmospheric LUT bake | Transmittance/multiple-scattering bake is dependency-invalidated outside the main pass list. Retain the existing reuse of transmittance inside multiscattering. Low priority versus PT texture/live-state work. [Host](../rdn/Renderer_Pipeline.cpp) |
| PT sky bake | With clouds enabled, only thread zero needs to write sun state; the full fixed dispatch otherwise returns. Use a minimal dispatch in that mode. For clear sky, avoid rebaking when all inputs truly remain unchanged. [Source](../shaders/Pass_pt_skybake_v8.hlsl) |
| Cloud noise bake | Dirty-gated procedural volume generation. Prioritize format/cache locality and avoiding unnecessary rebakes; it is not steady-state PT cost. [Source](../shaders/Pass_cumulus_noise_v8.hlsl) |
| Cloud density bake | Already uses camera-following tagged bricks and reuses matching bricks. Preserve identical shared boundary vertices; tune brick layout and update dispatch to newly exposed bricks. [Source](../shaders/Pass_cumulus_density_v8.hlsl) |
| Cloud ambient bake | Dirty-gated small table, 16-direction integration. Low steady-state priority. Reuse frame-uniform atmosphere/sun setup where possible. [Source](../shaders/Pass_cumulus_ambient_v8.hlsl) |
| Cloud light bake | Already shares one density evaluation per voxel column. Replace repeated summation above each cell with a segmented suffix scan; retain separate directional sunlight integration. [Source](../shaders/Pass_cumulus_light_v8.hlsl) |
| Cloud environment bake | Integrates a 2D environment each frame. Tile scheduling/temporal updates are candidates, but temporal subsampling changes responsiveness and needs quality measurements. [Source](../shaders/Pass_cumulus_environment_v8.hlsl) |
| Secondary clouds | Full-screen dispatch, only pixels with queued misses march. Compact sparse work when beneficial; shrink query hot/cold data; preserve reservoir normalization across multiple misses. [Source](../shaders/Pass_cumulus_secondary_v8.hlsl) |
| Primary atmosphere/clouds | Writes scattering/transmittance and clears guide slices. Avoid redundant clears when the guide producer definitely overwrites them. Separate FP32 radiance from bounded data before format reduction. [Source](../shaders/Pass_atmosphere_primary_v8.hlsl) |
| Cloud guides | Stable guide integration deliberately differs from jittered radiance integration. Share density data/hierarchy where possible; directly reusing noisy radiance samples would change denoiser behavior. [Source](../shaders/Pass_cumulus_guides_v8.hlsl) |
| Shading/composite | Includes extra transparent-guide tracing, not merely a fullscreen add. Separate the rare guide-through-glass path; make reference accumulation/debug albedo conditional. [Source](../shaders/Pass_shading_v8.hlsl) |
| DLSS / optional neural rendering | Most guide formats are already half/packed; depth correctly remains R32. Only allocate/write optional guides when supported and consumed. Validate guide conventions and temporal quality with each change. [Source](../rdn/PostProcess/DLSSManager.cpp) |
| Auto-exposure reduce | Lane zero serially sums 64 shared entries, then issues two global atomics per tile. Use wave reductions plus a small group combine; compare a two-level reduction if contention matters. [Source](../shaders/Pass_autoexpose_reduce_v8.hlsl) |
| Auto-exposure finalize | Tiny fixed dispatch; not a useful first memory target. Preserve valid-pixel weighting and adaptation semantics. [Source](../shaders/Pass_autoexpose_finalize_v8.hlsl) |
| Postprocess | Writes all six output slices, including retired/debug outputs. Execute only requested outputs, retaining editor views on demand. Profile sharpening neighborhood reads separately. [Source](../shaders/Pass_postprocess_v8.hlsl) |
| Host barriers | Active passes are separated by global UAV barriers. Build resource-level dependencies and scoped barriers; never remove the learning/cache producer-consumer fences blindly. [Source](../rdn/Renderer.cpp) |
| Geometry/AS | Fast-trace BLAS compaction and opaque/alpha splitting already exist. Explore separate position and shading-attribute streams, local index width and primitive ordering for cache locality; do not treat whole instance-record size as bytes fetched for each hit. [Source](../rdn/Renderer_Pipeline.cpp) |

Debug/reference work deserves a simple first experiment. SHADING_DEBUG_SLICES defaults to 1 and the host does not override it for shading/postprocess. The reference buffer alone has a logical 16-byte read, 16-byte write and later 16-byte read per pixel, approximately **94.9 MiB/frame at 1080p**, before debug output stores. The existing switch removes that traffic, but postprocess still writes six slices even when it is off. Preserve requested debug views through an explicit mode rather than deleting them. [Default](../shaders/Constants_v8.hlsli), [reference update](../shaders/Pass_shading_v8.hlsl), [output stores](../shaders/Pass_postprocess_v8.hlsl).

**9. One light sample per screen-space cell?**

Share a **proposal or candidate set**, not a final shaded result or visibility value.

A common cell can reuse a selected light identity while each receiver evaluates its own geometry, BSDF and visibility, with the exact marginal selection probability for the shared proposal. That can be unbiased but introduces correlation. If a cell mixes distinct surfaces, normals or depths, proposal quality can deteriorate sharply.

A better first experiment is shared/coherent upper-tree traversal or a small candidate pool, followed by per-pixel selection and exact PDFs. A screen grid should not replace the persistent world-space learned grid: it serves a different purpose and loses identity under camera motion. This is a secondary experiment after fixing texture access and measuring traversal traffic; it is not a substitute for material/live-state work.

**10. Verification and implementation order**

Fresh verification performed for this audit:

- Compiled the current production fast PT shader to SM 6.9 DXIL and inspected dynamic-resource handle annotations.
- Compiled and ran a headless D3D12 material diagnostic using the current production BSDF functions.
- Reproduced the reciprocal-attenuation defect and identical positive/negative anisotropy mapping.
- Reconciled current CPU allocations with shader layouts and active pass selection.
- Reviewed previous material corrections and performance work so fixed hemisphere folding, narrow-coat precision, dirty resolve and primary reservoir clears were not reported as new work.

Artifacts: [GPU results](../out/memory-material-audit/results.csv), [diagnostic shader](../out/memory-material-audit/bias.hlsl), [runner](../out/memory-material-audit/run.ps1), [current compiled PT](../out/memory-material-audit/pt-fast.ll). The analytic LUT fixture is explicitly not the renderer's generated LUT. No fresh whole-frame Nsight capture or city-scene A/B was taken; current L2 bandwidth, register allocation, occupancy and millisecond gains remain unmeasured.

Recommended delivery order:

1. **Correctness baseline:** divergent descriptors; delta/NEE bookkeeping; texture factors and import units; independent coverage/transmission; normal/tangent fixes.
2. **Low-risk traffic reductions:** on-demand debug outputs, mode-sized allocations, consumer-gated guide rays; texture cooking and footprint mips with visual comparisons.
3. **Material execution redesign:** common classes, prepared vertex state, unified sample/evaluation, explicit closure masks; backend register/live-state A/B.
4. **Physical closure upgrades:** reciprocal layer coupling, multiscattering, proper glass media and SSS boundaries; version old material appearances.
5. **Targeted cache/scheduling work:** hot/cold light/cache records, training-spill layout, upper-tree coherence, resource-scoped barriers and selective expensive-material queues.

Acceptance needs more than an image that looks cleaner. Add production-LUT white-furnace tests; reflection reciprocity and transmission eta-scaled adjoint checks; sampler/PDF histograms including null and delta mass; NEE on/off agreement with matching path budgets; nested glass and absorbing-emitter tests; tangent/normal tests under mirrored and nonuniform transforms; SSS boundary tests; and glTF fixtures sharing textures with different factors.

Performance comparisons should use fixed cameras and moving-camera routes, several material mixes, equal ray/sample budgets, and cold/warm caches. Record median and tail frame times plus image error, actual L2 traffic, registers and RT live state. Preserve all current material features; specialize their execution instead of quietly removing them.
