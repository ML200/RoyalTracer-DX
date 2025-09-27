# Royal Tracer DX
A real-time path tracer written in HLSL using the DirectX 12 API.

## Background
What started as a port of the [RoyalTracer university project](https://github.com/Royal-Project-Group/royaltracer) to DirectX quickly became a standalone rendering engine. In my [Bachelor's Thesis](https://ml200.github.io/university/2025/05/28/thesis.html), I implemented and optimized ReSTIR to enhance the renderers' real-time capabilities. Since then, the focus has shifted to implementing and evaluating state-of-the-art algorithms for improving unbiased sampling efficiency. For that reason, the renderer is still missing some core backend features such as texture maps, GLTF loading, or correct instancing.

## Features
<details>
  <summary><strong>Materials</strong></summary>
  - Simple two-lobe PBR material model
    - Lambertian Diffuse lobe
    - Energy-conserving GGX specular lobe
  - VNDF importance sampling
</details>
<details>
  <summary><strong>Light Sampling</strong></summary>
</details>
<details>
  <summary><strong>ReSTIR</strong></summary>
</details>
<details>
  <summary><strong>Pipeline</strong></summary>
</details>
<details>
  <summary><strong>Denoiser</strong></summary>
</details>

## Planned


## Setting up the project
### Prerequisites:
- A reasonably recent Windows 11 version, as this project uses the latest DirectX Agility SDK
- Install Visual Studio 2022 (build tools)

### Clion: 2024.3.2 or newer
- Set up the toolchain: Visual Studio (should be auto-detected, select it). Delete any other toolchain.
- Configure the CMAKE project: Select Visual Studio as the toolchain. The build directory should be named "cmake-build-debug-visual-studio" to exempt it from pushing to GitHub. Select "use default" for the generator and "Release" for build type.
- Delete the current cmake-build-debug-visual-studio directory if it exists
- Reload the CMAKE project (file -> reload CMAKE project)
- Build and run the project. Includes should be automatically included in the build directory.

### Visual Studio: 2022 or newer
- Open the project folder
- VS should automatically run CMAKE
- Run the program.
