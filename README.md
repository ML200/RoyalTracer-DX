# Royal Tracer DX
A real-time path tracer written in HLSL using the DirectX 12 API.

## Background
What started as a port of the [RoyalTracer university project](https://github.com/Royal-Project-Group/royaltracer) to DirectX quickly became a standalone rendering engine. In my [Bachelor's Thesis](https://ml200.github.io/university/2025/05/28/thesis.html), I implemented and optimized ReSTIR to enhance the renderers' real-time capabilities. Since then, the focus has shifted to implementing and evaluating state-of-the-art algorithms for improving unbiased sampling efficiency. For that reason, the renderer is still missing some core backend features such as texture maps, GLTF loading, or correct instancing.

## Features
<details>
  <summary><strong>Materials</strong></summary>

  <table>
    <tr>
      <!-- LEFT: text & code -->
      <td style="width:58%; vertical-align:top; padding-right:12px;">

        <!-- Overview -->
        <p><strong>Overview</strong><br>
        <!-- 1–2 sentences describing what this feature is and why it matters. -->
        </p>

        <!-- Key points -->
        <ul>
          <li><!-- Point 1: what it supports / how it works --></li>
          <li><!-- Point 2: implementation detail / benefit --></li>
          <li><!-- Point 3: when to use / edge cases --></li>
        </ul>

        <!-- Usage (replace with your actual flags/args) -->
        <p><strong>Usage</strong></p>
        <pre><code># CLI example
./pt --scene PATH/TO/SCENE.json --spp 512 --bounces 8 --device cuda
</code></pre>

        <!-- Tips / Notes -->
        <p><strong>Tips</strong></p>
        <ul>
          <li><!-- Tip 1: performance or quality knob --></li>
          <li><!-- Tip 2: common pitfall to avoid --></li>
        </ul>

      </td>

      <!-- RIGHT: image (diagram, render, or gif) -->
      <td style="vertical-align:top;">

        <p align="center" style="margin:0;">
          <!-- Use JPG/PNG for images; GIF/MP4 for comparisons (link to MP4 via thumbnail) -->
          <img src="assets/features/REPLACE_ME.png"
               alt="<!-- concise, descriptive alt text for accessibility -->"
               width="360">
          <br>
          <em><!-- Short caption explaining what the image shows --></em>
        </p>

        <!-- Optional: a second, smaller image or badge row
        <p align="center" style="margin-top:8px;">
          <img src="assets/features/REPLACE_SECOND.png" alt="alt text" width="160">
        </p>
        -->

      </td>
    </tr>
  </table>

  <!-- Optional: collapsible deep-dive inside the card -->
  <!--
  <details>
    <summary><em>Show implementation notes</em></summary>
    <ul>
      <li><!-- Low-level detail 1 --></li>
      <li><!-- Low-level detail 2 --></li>
    </ul>
    <pre><code>// Short code snippet (C++/CUDA/etc.)
</code></pre>
  </details>
  -->

</details>

- ReSTIR
- Pipeline
- Light Tree
- Denoiser

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
