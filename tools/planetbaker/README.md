# planet_bake

Standalone CUDA tool that simulates planetary geology, climate, and erosion, and bakes
the results to cubed-sphere textures (KTX2 + EXR sidecars) for a separate path-traced
planet renderer to consume.

This is a fully standalone project under `tools/planetbaker/`. It shares no headers
or types with the parent renderer. The only contract between them is the on-disk
texture file format.

See [PLAN.md](PLAN.md) for the full architecture and milestone roadmap.

## Status

M0 skeleton: window, ImGui, CUDA hello kernel. Nothing else implemented yet.

## Build

```
cmake -S . -B build -G Ninja
cmake --build build
```

CLion: open `tools/planetbaker/` directly.

Requirements:
- CUDA Toolkit 12.4+ (13.0 verified)
- CMake 3.27+
- MSVC 19.38+ (Windows), GCC 12+ or Clang 16+ (Linux)
- A CUDA-capable GPU

`CMAKE_CUDA_ARCHITECTURES` defaults to `native` (compiles for the local GPU only).
Override with e.g. `-DCMAKE_CUDA_ARCHITECTURES="80;86;89;90"` for a portable binary.

## Run

```
./build/planet_bake
```

Opens an ImGui window. Prints CUDA device info and runs a hello kernel on startup.

## Tests

```
ctest --test-dir build --output-on-failure
```
