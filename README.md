# Royal Tracer DX

A real-time path tracer written in HLSL using the DirectX 12 API.

![Monk statue in a cathedral scene with emissive light strips](media/monk.png)

## Background

What started as a port of the [RoyalTracer university project](https://github.com/Royal-Project-Group/royaltracer) to DirectX quickly became a standalone rendering engine. In my [Bachelor's Thesis](https://ml200.github.io/university/2025/05/28/thesis.html), I implemented and optimized ReSTIR to enhance the renderers' real-time capabilities. Since then, the focus has shifted to implementing and evaluating state-of-the-art algorithms for improving unbiased sampling efficiency.

## Features

### Model Loading
Supports OBJ and glTF/glB formats via [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader) and a custom tiny_gltf loader. Textures are loaded through stb_image with DDS decompression via DirectXTex. Models, materials, and per-instance transforms are unified into a single scene representation.

### Material Model
![Material comparison showing varying roughness and metallic properties](media/balls.png)

A four-lobe energy-conserving BXDF with layered evaluation:
- **Sheen** -- Charlie NDF for fabric-like surfaces
- **Clearcoat** -- Dielectric GGX with independent roughness and Fresnel
- **GGX Specular/Transmission** -- Anisotropic microfacet model with VNDF importance sampling, supporting reflection and refraction through nested dielectrics with per-bounce IOR stack tracking
- **Lambertian Diffuse** -- Cosine-weighted base layer

Supported material properties include roughness, metallic, anisotropy with rotation, clearcoat strength, sheen, transmission color, IOR, and alpha masking. Albedo, normal, and roughness/metallic/AO texture maps are supported with independent UV scaling.

### Light Sampling
![Many-light scene with hundreds of emissive spheres](media/LTree.png)

A light tree built over the scene's emissive triangles provides efficient importance sampling for many-light environments. The tree uses a TLAS/BLAS hierarchy with precomputed visibility cones and geometric importance weights (receiver cosine, distance attenuation) to guide traversal. This allows the renderer to handle scenes with hundreds of emissive primitives without per-light overhead.

### ReSTIR
![MIS convergence test with emissive spheres on reflective surfaces](media/MIS.png)

Full ReSTIR implementation for both direct (DI) and indirect (GI) illumination. Each path uses temporal and spatial reservoir resampling with pairwise MIS for unbiased combination of canonical and neighbor samples. M-capping limits temporal history length. A post-process boiling filter using groupshared edge-aware variance clamping suppresses temporal instability in low-sample regions.

### Rendering Pipeline
Compute-based wavefront path tracer using inline ray tracing. The pipeline is split into discrete passes:
1. **Raygen** -- Primary rays, multi-bounce path tracing with NEE
2. **Temporal DI/GI** -- Temporal reservoir resampling
3. **Spatial DI select + merge** -- Neighbor selection and spatial resampling for DI
4. **Spatial GI select + merge** -- Neighbor selection and spatial resampling for GI
5. **Boiling filter** -- Temporal stability pass for DI and GI
6. **Shading** -- Final accumulation, motion vectors, and DLSS input preparation
7. **Post-process** -- Tone mapping (PBR Neutral) and sRGB gamma correction

Pixel indexing uses 4x8 tile swizzling for improved cache coherence during spatial reuse.

### Denoiser
NVIDIA DLSS Ray Reconstruction is used for denoising. The shading pass provides linear depth, geometric normals, diffuse and specular albedo, surface and specular motion vectors, roughness, specular hit distance, and a disocclusion bias hint. Emitter and sky pixels are routed through the diffuse denoiser channel with clamped radiance to prevent ghosting.

### Procedural Sky
Physically-based atmosphere rendering with single-scattering ray marching (Rayleigh + Mie + ozone absorption). Uses a Cornette-Shanks phase function for Mie scattering. The sun position is computed from latitude, longitude, day-of-year, and UTC time. Runtime-editable parameters include turbidity, sun/sky intensity, and simulation speed with automatic night speedup.

### Volumetric Clouds
Procedural volumetric clouds using a Nubis-style ray march with stochastic sampling. Two-lobe Cornette-Shanks Mie phase function for cloud lighting. Integrated with the path tracer's per-pixel random seed for DLSS-compatible temporal integration.

## Setting up the project

### Requirements
- Windows 11 (recent version for DirectX Agility SDK support)
- NVIDIA RTX 40-series GPU or newer
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
