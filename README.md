# Royal Tracer DX

A real-time path tracer written in HLSL using the DirectX 12 API.

![Bistro exterior scene](media/bistro_clean.png)

## Background

What started as a port of the [RoyalTracer university project](https://github.com/Royal-Project-Group/royaltracer) to DirectX quickly became a standalone rendering engine. In my [Bachelor's Thesis](https://ml200.github.io/university/2025/05/28/thesis.html), I implemented and optimized ReSTIR to enhance the renderers' real-time capabilities. Since then, the focus has shifted to implementing and evaluating state-of-the-art algorithms for improving unbiased sampling efficiency.

![Sponza interior](media/sponza_clean.png)

## Features

### Model Loading
Supports OBJ and glTF/glB formats via [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) and [tinygltf](https://github.com/syoyo/tinygltf). Textures are loaded through stb_image with DDS decompression via [DirectXTex](https://github.com/microsoft/DirectXTex). Models, materials, and per-instance transforms are unified into a single scene representation.

### Material Model

<p float="left">
  <img src="media/dragon.png" width="24%" />
  <img src="media/sheen_clean.png" width="24%" />
  <img src="media/metal_clean.png" width="24%" />
  <img src="media/clearcoat_clean.png" width="24%" />
</p>

![Material model layer diagram](media/material_model.svg)

A four-lobe energy-conserving BXDF with layered evaluation:
- **Sheen** -- [Charlie NDF](https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_sheen.pdf) for fabric-like surfaces
- **Clearcoat** -- Dielectric GGX with independent roughness and Fresnel
- **[GGX](https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf) Specular/Transmission** -- Anisotropic microfacet model with VNDF importance sampling, supporting reflection and refraction through nested dielectrics with per-bounce IOR stack tracking
- **Lambertian Diffuse** -- Cosine-weighted base layer

### Light Sampling

<video src="media/tlas_refit.mp4" controls muted loop width="100%">Your browser does not support the video tag.</video>

A [light tree](https://fpsunflower.github.io/ckulla/data/many-lights-hpg2018.pdf) built over the scene's emissive triangles provides efficient importance sampling for many-light environments. The tree uses a TLAS/BLAS hierarchy with precomputed visibility cones and geometric importance weights (receiver cosine, distance attenuation) to guide traversal. This allows the renderer to handle scenes with hundreds of emissive primitives without per-light overhead.

Light tree builds on the CPU. When lights move or their brightness changes, the tree is refit/rebuilt asynchronously -- the clip above shows TLAS refit keeping pace with animated emitters.

### ReSTIR

<p float="left">
  <img src="media/twr_norestir.png" width="32%" />
  <img src="media/twr_restir.png" width="32%" />
  <img src="media/twr_denoised.png" width="32%" />
</p>

*Left: path tracing, no ReSTIR. Middle: ReSTIR (raw 1 spp). Right: ReSTIR + DLSS Ray Reconstruction.*

Unbiased [ReSTIR DI/PT](https://research.nvidia.com/publication/2022-07_generalized-resampled-importance-sampling-foundations-restir) (reconnection shift only) on a **unified DI/GI reservoir**: NEE, environment miss, and path-integrand candidates all feed one reservoir stream, with sentinel matIDs discriminating direct samples from indirect ones. Each path uses temporal and spatial reservoir resampling with [pairwise MIS](https://intro-to-restir.cwyman.org/) for unbiased combination of canonical and neighbor samples. Temporal permutation sampling decorrelates reuse patterns across frames, and the temporal cCap is modulated by a per-pixel **duplication map** so highly-shared samples refresh quickly instead of persisting as fireflies (Lin et al. 2026 §5).

### Rendering Pipeline
![Render pipeline diagram](media/pipeline.svg)

The path tracer is using new HitObj ray tracing with Shader Execution Reordering (SER) for wavefront-like coherence without an explicit wavefront architecture. The pipeline is split into discrete passes:
1. **Raygen** -- Primary rays, multi-bounce path tracing with NEE. All candidates (NEE, env miss, path integrand) are written into a single **unified DI/GI reservoir**, keyed by sentinel matIDs. Thanks to SER, aggressive russian roulette sampling allows for 30+ bounces with barely any performance impact
2. **Temporal reuse** -- Pairwise-MIS temporal resampling on the unified reservoir. Permutation sampling breaks up temporal correlations that become very apparent in the denoiser; the temporal cCap is adaptively lowered where the previous frame's duplication map shows high sample reuse
3. **Reuse-texture partner select** -- A stack of three precomputed self-inverting **reuse textures** (Lin, Kettunen, Wyman 2026 §3) gives every pixel a guaranteed-symmetric spatial partner in a single texture load, skipping the usual neighbor-search pass
4. **Spatial reuse** -- Pairwise-MIS reconnection-shift merge of the canonical reservoir with its paired partner, outputting combined DI+GI radiance
5. **Duplication map** -- Compute pass that scans each pixel's 17×17 neighborhood and counts matches of the packed V2 (reconnection-vertex) identifier. Reconnection and hybrid shifts preserve V2 bit-for-bit, so a matching V2 in a neighbor reliably signals a shifted copy of the same initial candidate; the count normalized to D ∈ [0, 1] measures how far that sample has spread. Next frame's temporal reuse reads D at the permuted-neighbor coord and shrinks the effective temporal M-cap toward 1 via `lerp(tempMcap, 1, pow(D, 0.1))` -- the low exponent ramps aggressively even at small D, refreshing the chain before a firefly can persist. Environment-miss samples skip the reduction (their V2 is a pixel-unique synthetic discriminator and their bounded radiance can't amplify, so capping there would only defeat sky accumulation). Implementation uses a 32×32 groupshared cache so the 288 per-thread reads collapse to 4 cooperative loads
6. **Shading** -- Final accumulation, motion vectors, and DLSS input preparation
7. **Post-process** -- Tone mapping (PBR Neutral) and sRGB gamma correction (after denoising)

### Opacity Micro-Maps
Alpha-tested geometry (foliage, fences, etc.) uses [Opacity Micro-Maps](https://github.com/NVIDIA-RTX/OMM) (OMMs) built with the NVIDIA OMM SDK. OMMs encode per-microtriangle opacity into the BVH, allowing the hardware to skip transparent regions during traversal without invoking any-hit shaders, significantly improving ray tracing performance on scenes with heavy alpha-tested content.

### Denoiser
NVIDIA DLSS Ray Reconstruction is used for denoising using NVIDIA [Streamline](https://github.com/NVIDIA-RTX/Streamline). On supported GPUs, DLSS frame generation can be used to improve performance.

## Setting up the project

### Requirements
- Windows 11 (recent version for DirectX Agility SDK support)
- NVIDIA RTX 40-series GPU or newer (can possibly run on older HW as well but no guarantee it works due to frame gen)
- Visual Studio 2022 (build tools)

### CLion (2024.3.2 or newer)
1. Set up the toolchain: select Visual Studio (should be auto-detected). Delete any other toolchain.
2. Configure the CMake project: select Visual Studio as the toolchain. Name the build directory `cmake-build-debug-visual-studio`. Select "use default" for the generator and "Release" for build type.
3. Delete the existing `cmake-build-debug-visual-studio` directory if it exists.
4. Reload the CMake project (File > Reload CMake Project).
5. Build and run. Includes are automatically placed in the build directory.

### Visual Studio (2022 or newer)
1. Open the project folder.
2. VS should automatically run CMake.
3. Build and run.
