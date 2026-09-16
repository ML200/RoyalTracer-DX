# Royal Tracer DX

![License](https://img.shields.io/badge/license-MIT-blue)
![Platform](https://img.shields.io/badge/platform-Windows%2011-lightgrey)
![API](https://img.shields.io/badge/API-DirectX%2012-76b900)
![Language](https://img.shields.io/badge/HLSL%20%7C%20C%2B%2B-informational)

Royal Tracer DX learns which lights illuminate a surface and which directions carry indirect light. It combines that sampling with a world-space radiance cache and ReSTIR spatial reuse in a real-time DirectX 12 path tracer.

![Studio interior](media/studio1.webp)

## Rendering pipeline

![SHARC trains and resolves the cache. Path tracing uses learned light cuts, path guiding, and cached radiance. ReSTIR spatial reuse combines samples through texture-defined pixel pairs.](media/pipeline.svg)

The regular path tracer has three main stages:

1. **SHARC:** Trace a rotating subset of pixels to update cached illumination and train path guiding. Resolve new observations into persistent history.
2. **Path tracing:** Sample lights through the light tree and learned cuts. Mix guided directions with material sampling, and query SHARC when the path has spread sufficiently.
3. **ReSTIR spatial reuse:** Reconnect samples from paired pixels, evaluate visibility, and combine them with pairwise multiple importance sampling (MIS).

Training and rendering share the same light sampler and material evaluation. The training paths also fit the guide used by subsequent paths, so improvements in emitter selection feed both the cache and directional sampling.

### Light trees and learning light cuts

A bright emitter behind a wall contributes nothing to direct illumination at the shaded point. Its power and distance alone can still make it an attractive sample. Learning visibility helps avoid spending shadow rays on that emitter repeatedly.

The [light tree](https://fpsunflower.github.io/ckulla/data/many-lights-hpg2018.pdf) organizes emissive triangles in a two-level hierarchy over instances and their geometry. Traversal estimates each cluster's importance from emitted power, distance, and orientation. This selects a light without evaluating every emitter at every surface.

[Learned light cuts](https://kevincosner.github.io/publications/Wang2021LCR/index.html) refine that hierarchy's sampling distribution using measured contributions, including occlusion. Each spatial cell maintains a cut of up to 32 clusters. Learning splits useful clusters and merges less useful ones, allocating detail where light selection benefits. Cells distinguish surface orientation and retain their history as the camera moves. Rendering and SHARC training both use the learned sampler.

### SHARC

![Interior rendered with SHARC](media/cache1.webp)

*SHARC example: interior illumination.*

[Spatially Hashed Radiance Caching](https://github.com/NVIDIA-RTX/SHARC) reuses indirect illumination between nearby surface points. A sparse hash grid stores estimates from training paths and accumulates them over time. Grid spacing grows with distance from the camera, concentrating detail near the viewer.

Direct lighting at camera-visible surfaces remains traced. At eligible secondary surfaces, the cache supplies the diffuse and rough reflection contribution, including further bounces. Sharper specular components can continue tracing independently. Material color is factored out during storage and restored at the query point, reducing the color variation the cache must represent.

Queries check surface identity, normal and plane compatibility, history confidence, and the path's spatial footprint. The footprint check delays reuse until scattering has spread the path beyond the cache's spatial resolution. Rejected queries continue tracing. Cache reuse trades tracing cost for approximation error; cold regions and changing illumination need new observations before their estimates become reliable.

### Path guiding

<table>
  <tr>
    <th width="50%">Path guiding off</th>
    <th width="50%">Path guiding on</th>
  </tr>
  <tr>
    <td valign="top"><a href="media/pg2.webp"><img src="media/pg2.webp" alt="Path guiding disabled" width="100%" /></a></td>
    <td valign="top"><a href="media/pg1.webp"><img src="media/pg1.webp" alt="Path guiding enabled" width="100%" /></a></td>
  </tr>
</table>

Material sampling describes how a surface scatters light. It does not know that most incoming illumination may arrive through a small doorway or from a brightly lit wall. [Path guiding](https://jannovak.info/publications/PathGuide/index.html) learns this directional structure from traced paths.

The implementation fits up to eight directional lobes per spatial cell using the sparse SHARC training paths. Each lobe records a direction, angular extent, weight, and distance estimate. The distance estimate adjusts the lobe for parallax at the actual shading point. Stale lobes lose influence, and cells without a usable local guide can fall back to a coarser cell.

Guided directions are mixed with the material's diffuse and rough reflection sampling. Path weights use the matching mixture probability. Guiding changes which paths are traced; SHARC supplies an approximate continuation when a cache query succeeds. They complement each other wherever the cache rejects a query or still needs training.

### Optimized ReSTIR spatial reuse

The regular pipeline resamples diffuse and rough reflection contributions using compact, 32-byte reservoirs. A reservoir retains one selected sample and the weights needed to combine it with other samples.

The central optimization is reciprocal neighbor selection, following [ReSTIR PT Enhanced](https://research.nvidia.com/labs/rtr/publication/lin2026restirptenhanced/). Precomputed reuse textures pair pixels symmetrically: if A selects B, B selects A. The shift pass evaluates each neighbor's sample at the receiving surface and stores its contribution and visibility. The merge pass then reads both sides of the pair for MIS, reusing those evaluations.

Material and geometric compatibility checks reject unsuitable pairs. Repeated samples within a pixel reuse their visibility result, and the merge loads the full neighbor payload only when that sample wins selection. These changes reduce redundant ray queries and memory traffic. Spatial reuse stays within the current frame; SHARC and the learned samplers maintain their own history.

The ReSTIR PT hybrid shift mode is deprecated.

### Technology comparison

The captures add features cumulatively. Select an image to view it at full resolution.

<table>
  <tr>
    <th width="50%">1 · Base path tracing</th>
    <th width="50%">2 · + Learned light cuts</th>
  </tr>
  <tr>
    <td><a href="media/comp1.webp"><img src="media/comp1.webp" alt="Base path tracing" width="100%" /></a></td>
    <td><a href="media/comp2.webp"><img src="media/comp2.webp" alt="Base path tracing with learned light cuts" width="100%" /></a></td>
  </tr>
  <tr>
    <th>3 · + SHARC</th>
    <th>4 · + Path guiding and ReSTIR</th>
  </tr>
  <tr>
    <td><a href="media/comp3.webp"><img src="media/comp3.webp" alt="Learned light cuts with SHARC" width="100%" /></a></td>
    <td><a href="media/comp4.webp"><img src="media/comp4.webp" alt="Learned light cuts, SHARC, path guiding, and ReSTIR spatial reuse" width="100%" /></a></td>
  </tr>
</table>

| Configuration | What changes |
| --- | --- |
| Base path tracing | Emitter selection uses geometric importance. Indirect paths use material sampling. |
| + Learned light cuts | Observed visibility and contribution refine emitter selection. |
| + SHARC | Eligible indirect continuations use accumulated illumination, introducing cache approximation. |
| + Path guiding and ReSTIR spatial reuse | Guided continuations favor learned directions. Reciprocal pixel pairs share samples and reuse their evaluations. |

## Material model

<table>
  <tr>
    <td width="20%" align="center" valign="middle"><a href="media/dragon.png"><img src="media/dragon.png" alt="Dragon material" width="100%" /></a></td>
    <td width="20%" align="center" valign="middle"><a href="media/sheen_clean.png"><img src="media/sheen_clean.png" alt="Sheen material" width="100%" /></a></td>
    <td width="20%" align="center" valign="middle"><a href="media/metal_clean.png"><img src="media/metal_clean.png" alt="Metal material" width="100%" /></a></td>
    <td width="20%" align="center" valign="middle"><a href="media/clearcoat_clean.png"><img src="media/clearcoat_clean.png" alt="Clearcoat material" width="100%" /></a></td>
    <td width="20%" align="center" valign="middle"><a href="media/sss1.webp"><img src="media/sss1.webp" alt="Dragon with subsurface scattering" width="100%" /></a></td>
  </tr>
  <tr>
    <td align="center">Dragon</td>
    <td align="center">Sheen</td>
    <td align="center">Metal</td>
    <td align="center">Clearcoat</td>
    <td align="center">SSS</td>
  </tr>
</table>

*Material examples, including subsurface scattering (SSS). Select an image for full resolution.*

![Layered scattering: sheen, clearcoat, GGX reflection and transmission, then diffuse. Emission is separate.](media/material_model.svg)

The material model evaluates four scattering lobes in order: [Charlie sheen](https://blog.selfshadow.com/publications/s2017-shading-course/imageworks/s2017_pbs_imageworks_sheen.pdf), dielectric clearcoat, anisotropic [GGX reflection and transmission](https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf), and Lambertian diffuse. Each layer attenuates the contribution beneath it. Clearcoat has its own roughness, while nested dielectrics track refractive-index transitions along the path. Emission is evaluated separately.

## Scene support

OBJ and glTF/GLB scenes support textured materials and instancing. Opacity micromaps accelerate alpha-tested geometry. NVIDIA DLSS Ray Reconstruction provides denoising and upscaling.

### Minecraft chunks

Minecraft chunk rendering is supported.

![Minecraft city rendered from chunks](media/minecraft1.webp)

## Build

Requires Windows 11, an NVIDIA RTX 40 series GPU or newer, Visual Studio 2022 with C++ build tools and Windows SDK 10.0.22621.0, and CMake 3.25.2 or newer.

Run from the repository root:

```powershell
cmake -S Pathtracer -B Pathtracer/build -G "Visual Studio 17 2022" -A x64
cmake --build Pathtracer/build --config Release
```

## Controls

| Input | Action |
| --- | --- |
| W / A / S / D | Move |
| Space / Left Ctrl | Ascend / descend |
| Left mouse drag | Look around |

## Background and credits

The renderer began as a DirectX port of the [RoyalTracer university project](https://github.com/Royal-Project-Group/royaltracer). Its ReSTIR work is documented in the [2025 bachelor's thesis](https://ml200.github.io/university/2025/05/28/thesis.html).

Scenes include [Amazon Lumberyard Bistro](https://developer.nvidia.com/orca/amazon-lumberyard-bistro) and Crytek Sponza. The project uses [Streamline](https://github.com/NVIDIA-RTX/Streamline), [Opacity Micro-Map SDK](https://github.com/NVIDIA-RTX/OMM), [tinyobjloader](https://github.com/tinyobjloader/tinyobjloader), [tinygltf](https://github.com/syoyo/tinygltf), [stb_image](https://github.com/nothings/stb), [DirectXTex](https://github.com/microsoft/DirectXTex), and [Dear ImGui](https://github.com/ocornut/imgui).

## Gallery

![Bistro exterior](media/bistro1.webp)

*Bistro.*

![Sponza interior](media/sponza1.webp)

*Sponza.*

<table>
  <tr>
    <td width="50%"><a href="media/harbor2.webp"><img src="media/harbor2.webp" alt="Harbor panorama" width="100%" /></a></td>
    <td width="50%"><a href="media/harbor3.webp"><img src="media/harbor3.webp" alt="Harbor at night" width="100%" /></a></td>
  </tr>
  <tr>
    <td align="center">Harbor</td>
    <td align="center">Harbor at night</td>
  </tr>
  <tr>
    <td><a href="media/ship2.webp"><img src="media/ship2.webp" alt="Ship in the harbor" width="100%" /></a></td>
    <td><a href="media/skyline.webp"><img src="media/skyline.webp" alt="City skyline" width="100%" /></a></td>
  </tr>
  <tr>
    <td align="center">Ship</td>
    <td align="center">Skyline</td>
  </tr>
</table>

![Island scene](media/island1.webp)

*Island.*
