# DLSS 5 Neural Rendering in RoyalTracer

Verified on 9 September 2026. The engine now has an **experimental native DLSS 5 NR backend**, disabled by default in the editor. It uses the user's unmodified, signed `nvngx_dlssnr.dll` 310.8.0.0. SR, RR and frame generation continue to use the public Streamline SDK.

## What NVIDIA currently publishes

| Component | Checked version | NR availability |
| --- | --- | --- |
| Public Streamline SDK | 2.14.1, released 8 September 2026 | Declares feature 1004 and uplift input/output/control-mask tags. The release does **not** provide an NR options header, integration guide, plugin or model runtime. |
| DLSS runtimes in that SDK | 310.9.1 | SR, RR and FG runtimes; no NR runtime. |
| Public NGX headers | Included with Streamline 2.14.1; public NVIDIA/DLSS headers also checked | Feature 18 is still `NVSDK_NGX_Feature_Reserved18`, without a public NR creation/evaluation contract. |
| User's `DLSS310.8.0-Streamline2.13` folder | Binary runtime bundle | Contains signed `sl.dlss_nr.dll` and `nvngx_dlssnr.dll`, but no SDK headers. |
| User's RenoDX addon | `renodx-dlss5.addon64` | A ReShade addon, not a library exposing a native engine interface. |

The supplied Streamline NR plugin was probed on the RTX 5090: SR/RR support checks succeeded, but NR reported `eErrorFeatureNotSupported`; its options call reported `eErrorFeatureMissing`. Merely adding feature 1004 to `featuresToLoad`, or passing reserved feature 18 through the ordinary NGX core, is insufficient here.

Primary references: [Streamline 2.14.1 release](https://github.com/NVIDIA-RTX/Streamline/releases/tag/v2.14.1), [public headers](https://github.com/NVIDIA-RTX/Streamline/tree/v2.14.1/include), [public NGX feature definitions](https://github.com/NVIDIA/DLSS/blob/main/include/nvsdk_ngx_defs.h), and [NVIDIA's DLSS 5 research](https://research.nvidia.com/labs/adlr/DLSS5/).

## Native adapter

`rdn/PostProcess/DLSSNRBridge.cpp` implements the adapter from inspected interface facts. The supplied binaries and the [OptiScaler DLSSNR investigation](https://github.com/Dagherbou/OptiScaler_DLSSNR/tree/dlss-neural-rendering/OptiScaler/dlssnr) were references; OptiScaler implementation code was not copied.

The adapter loads the NR model DLL directly and calls its private `NVSDK_NGX_D3D12_*` exports: `Init_Ext`, `PopulateParameters_Impl`, `CreateFeature`, `EvaluateFeature`, `ReleaseFeature` and `Shutdown1`. It implements the public `NVSDK_NGX_Parameter` interface with owned, typed storage. It neither obtains nor modifies Streamline's NGX parameter objects, and it does not initialize or shut down Streamline's NGX core.

The private initialization signature takes `(applicationId, dataPath, device, version, parameters)`: the application ID is 0, version is `0x15`, and creation uses feature 18. These facts apply only to the pinned runtime. No public SDK NR struct or function signature is assumed.

The bridge is named `nvngx.dll_royaltracer_nr.dll` because this runtime checks that its caller module's filename contains `nvngx.dll`. Calls remain inside this bridge, including in optimized builds. It is a separate engine library, with no hooks, binary patches, driver changes, ReShade dependency or borrowed game identity.

Both CMake staging and the bridge itself require this exact SHA-256 before loading:

```text
e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e
```

The editor also displays the file version and Authenticode result. A different runtime requires rechecking its ABI and passing the GPU tests before updating the pin. The bridge is intended for one renderer/context on one native D3D12 device, used from the render thread.

## Frame contract and synchronization

NR runs after successful RR reconstruction, AgX tone mapping and the final scene post-processing, before ImGui and FG's HUD-less capture. It only processes the reconstructed scene slice (layer 1); diagnostic slices bypass it and reset NR history.

| Input | Engine contract |
| --- | --- |
| Color | Display-resolution, display-encoded SDR `R8G8B8A8_UNORM`, **alpha 1**, copied out of the scene array into a dedicated 2D texture. |
| Output | Separate display-resolution `R8G8B8A8_UNORM` UAV; never aliases the input or backbuffer. |
| Depth | Render-resolution `R32_FLOAT`, reverse Z; `DLSSNR.DepthInverted = 1`. |
| Motion | Render-resolution `R16G16_FLOAT`, current-to-previous displacement, including camera motion, excluding jitter, in render-pixel units. |
| Subrects | Base `(0,0)`; color/output use display dimensions, depth/motion use their actual render dimensions. |
| Motion scales | **1.0 on both axes.** The private API takes guide-pixel vectors; it handles the differing subrect dimensions internally. |

The motion scaling was traced through the supplied NVIDIA plugin: at image VA `0x18003e8da` onward, it multiplies Streamline's normalized `mvecScale` by the motion-vector extent and stores the result in the backend frame's `+0xf0/+0xf4`; the backend writes these to `DLSSNR.MVecScaleX/Y` at `0x18003b38c` onward. RoyalTracer supplies Streamline with `1/renderWidth, 1/renderHeight`, so the corresponding native values are 1. Scaling again by display/render resolution would double-count that conversion.

The manager transitions color to shader-read, output to UAV and both guides from UAV to shader-read. It restores the guides on both success and failure and leaves a successful output in `COPY_SOURCE`. The renderer rebinds its descriptor heaps after every attempt.

Feature creation can record GPU work: the creation frame presents the original scene, submits normally, and completes its fence before the first evaluation on the next frame. `PrepareFrameGPUIdle()` runs after the existing previous-frame fence, including Present. Model release/recreation for tuning, disabling, resizing and shutdown occurs only with the GPU idle. Native interfaces returned by Streamline are held with their correct COM reference ownership.

NR inherits RR's actual history-reset signal, including preset changes, and skips failed or skipped RR frames. It also resets for camera cuts, guide/output size changes, bypassed views, manual reset and re-enabling. Tuning changes recreate the feature because this runtime reads those controls at creation. A runtime error latches failure and keeps the original scene visible; changing settings or disabling and re-enabling retries initialization.

Frame generation receives a persistent 2D copy of the displayed scene **including NR, before ImGui** as `HUDLessColor`. The backbuffer containing ImGui is tagged separately. Both remain valid through Present, and shutdown drains GPU work before releasing them.

## Build and enable

The existing Visual Studio build directory has already been configured with the supplied NR runtime. To reproduce from a developer shell with the project's existing CUDA/SDK prerequisites:

```powershell
cmake -S . -B cmake-build-relwithdebinfo-visual-studio `
  -DPATHTRACER_ENABLE_DLSSNR=ON `
  "-DDLSSNR_RUNTIME_FILE=C:/Users/Malte/Downloads/DLSS310.8.0-Streamline2.13/nvngx_dlssnr.dll"
cmake --build cmake-build-relwithdebinfo-visual-studio --target Pathtracer --parallel 6
```

The build stages the model at `<executable directory>/dlssnr/nvngx_dlssnr.dll`. It does not add the proprietary model to the source tree or download it. Public Streamline headers, import library and release runtimes have been updated together to 2.14.1. Runtime staging happens **after** copying `include/`, which contains older DLLs that would otherwise overwrite the selected SDK.

Launch `cmake-build-relwithdebinfo-visual-studio/Pathtracer.exe`, select the reconstructed scene, then open **View > DLSS 5 Neural Rendering** and check **Enable**. After model preparation, the successful-evaluation counter should increase. NR is opt-in each launch.

The panel includes these two independent selectors, which can also be set before enabling NR:

| Control | Choices | Runtime parameter |
| --- | --- | --- |
| Model style | Default (0), Natural (1), Cinematic (2) | `DLSSNR.Style` |
| Render preset | Default (0), Preset 1, Preset 2, Preset 3 | `DLSSNR.Hint.Render.Preset` |

These labels and ranges match the supplied RenoDX addon and the [reference NR controls](https://github.com/DaniilSokolyuk/video2dlssnr/blob/main/README.md#neural-rendering). They select the NR runtime's style and embedded-model preset request; they do not select another DLL or change RR's A–O presets. Intensity and local tone/structure remain independently adjustable. Each selection change recreates the feature after the GPU fence and resets history, including when returning to Default. Invalid programmatic selection values normalize to Default; the bridge also rejects out-of-range values.

The pinned runtime explicitly contains fallback behavior for unavailable presets. All 12 style/preset combinations completed creation and GPU evaluation in the regression. The three styles produced different output, while all four preset requests produced the same output within each style on the synthetic test frame. The numbered choices therefore must not be presented as four verified distinct networks or quality levels.

Alternative runtime discovery uses `DLSSNR_RUNTIME_PATH` (a file or directory, authoritative when set), then `<exe>/dlssnr/nvngx_dlssnr.dll`, then `<exe>/nvngx_dlssnr.dll`. Restart after changing the runtime location. `PATHTRACER_ENABLE_DLSSNR=OFF` builds the inactive backend without importing the bridge.

## Verification and limits

Run the standalone integration regression without building the CUDA renderer:

```powershell
./tests/run_dlssnr_tests.ps1 -RuntimePath "C:/Users/Malte/Downloads/DLSS310.8.0-Streamline2.13/nvngx_dlssnr.dll"
```

The test uses the actual managers and optimized bridge, submits D3D12 commands, waits for completion and reads back NR output. On an RTX 5090 with driver 610.88 it passed with D3D12 debug validation enabled: disabled-backend linkage, SR/RR initialization alongside NR, creation/evaluation, nonzero motion guides, all model-style/preset combinations, exact default restoration, deterministic history reset, tuning recreation, selection while disabled, retry after changing a failed selection, off/on, output and guide resizing, invalid dimensions, missing runtime and runtime-hash rejection. Fallback output is checked against the original constant-color pixels.

An additional sequence runs the engine's actual `DLSSManager` with preset F at Quality mode, converts its reconstructed output into opaque SDR, and evaluates NR on the **same command list**. This passed across temporal frames and an RR reset. RR evaluation also passed after NR context shutdown on the same device, verifying that NR teardown leaves RR operational.

The complete engine build and the modified post-process shader compile also passed. Existing unrelated compiler warnings remain. The synthetic GPU test establishes API/resource/lifetime behavior; it does not establish visual quality on moving production scenes, production-scene performance, or NR-plus-frame-generation presentation behavior. Live engine UI inspection was blocked because computer-use access to Pathtracer was not approved, so those visual checks remain outstanding.

This is a version-pinned experimental route, not an officially documented DLSS 5 engine integration. When NVIDIA publishes the full NR API and compatible plugin, replace the bridge with that interface while preserving the frame contract, reset propagation and fence discipline above.
