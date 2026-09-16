# Planet Texture Baker, Build Plan

A standalone CUDA C++ tool that simulates planetary geology, climate, and erosion, then bakes the results to cubed-sphere textures for a separate path-traced planet renderer to consume. This document is the implementation plan. Build in the order given.

## 1. Goals

1. Simulate a rocky/desert Earth-sized planet with tectonics, impacts, climate, erosion, clouds.
2. Bake results to a stack of cubed-sphere textures (KTX2 + EXR sidecars).
3. Provide an interactive debug viewer to inspect every intermediate field on a 3D globe before approving the bake.
4. Expose every parameter for tuning, both through a JSON config file and through live ImGui controls.
5. Architect the state, passes, and outputs as a superset that covers wet and lush worlds later, with only those passes stubbed for now.
6. Be entirely standalone. Lives in a subfolder of the path tracer repo, has its own build, shares no headers or types with the renderer. The only contract between them is the on-disk texture file format.

## 2. Constraints

- C++20 host code. CUDA 12+ device code. No other GPU backends.
- CMake build, fully self-contained (CPM or FetchContent for deps).
- Windows and Linux must both build cleanly. macOS is out of scope (no CUDA).
- No reference to anything in the parent path tracer repo. Treat it as if it does not exist.
- All output files written under an output directory chosen at runtime. Default `./out/`.
- Working precision is FP32 for simulation. Output is BC6H / BC7 / BC5 inside KTX2 plus EXR debug copies.
- Single-GPU. Assume CUDA device 0 unless overridden.

## 3. Repository layout

Place the entire tool under `planet_bake/` in the parent repo. Internal layout:

```
planet_bake/
  CMakeLists.txt
  README.md
  PLAN.md                  // this document
  cmake/
    FindCUDAToolkit.cmake  // only if needed
    cpm.cmake
  src/
    main.cpp
    app/
      viewer.h / .cpp
      ui.h / .cpp
      config.h / .cpp
      input.h / .cpp
    core/
      cubed_sphere.h / .cpp / .cu
      neighbor_table.h / .cpp
      planet_state.h / .cpp
      field.h / .cpp
      pass.h
      pipeline.h / .cpp
      cache.h / .cpp
      hash.h / .cpp
      log.h / .cpp
    passes/
      tectonics.h / .cu
      hex_grid.h / .cu
      impacts.h / .cu
      climate.h / .cu
      shallow_water.h / .cu
      erosion_thermal.h / .cu
      erosion_aeolian.h / .cu
      erosion_glacial.h / .cu
      erosion_hydraulic.h / .cu
      weathering.h / .cu           // stub for now
      vegetation.h / .cu           // stub for now
      biome_classifier.h / .cu
      bake.h / .cu
    io/
      ktx2_writer.h / .cpp
      exr_writer.h / .cpp
      bc_encoder.h / .cu
      config_io.h / .cpp
      cache_io.h / .cpp
    gpu/
      cuda_utils.h
      cuda_check.h
      cuda_gl_interop.h / .cpp
      memory.h / .cpp
      kernels_common.cuh
    math/
      vec.h
      quat.h
      sphere_math.h
    render/
      gl_loader.h / .cpp
      sphere_mesh.h / .cpp
      shader.h / .cpp
      camera.h / .cpp
      colormap.h / .cpp
  shaders/
    sphere.vert
    sphere.frag
    arrows.vert
    arrows.frag
  presets/
    dry_rocky.json
    wet_rocky.json           // placeholder, mostly defaults
    lush.json                // placeholder
  third_party/
    (CPM-fetched, do not commit)
  tests/
    cubed_sphere_test.cpp
    neighbor_table_test.cpp
    pass_pipeline_test.cpp
  out/                       // gitignored, runtime output
  cache/                     // gitignored, pass cache
```

## 4. Tech stack

Pin exact versions in CMake via CPM.

- CUDA Toolkit 12.4 or newer.
- C++20, MSVC 19.38+ on Windows, GCC 12+ or Clang 16+ on Linux.
- CMake 3.27+ with first-class CUDA language support.
- GLFW 3.4 for windowing.
- glad 2 for OpenGL 4.5 core function loading.
- Dear ImGui, docking branch, v1.90+.
- ImGuizmo for 3D widget overlays (optional, nice for picking).
- glm 1.0+ for math on the host. Device math is hand-rolled in `math/`.
- nlohmann/json 3.11+ for config I/O.
- libktx (KTX-Software) 4.3+ for KTX2 writing, including BasisU transcoder linkage off, raw BCn on.
- tinyexr 1.0+ for EXR writing.
- stb_image_write for PNG dumps (debug only).
- doctest for unit tests.

No Python, no shell scripts, no external tools required to run.

## 5. Build system

Single top-level `CMakeLists.txt` in `planet_bake/`. Targets:

- `planet_bake` executable, the main app.
- `planet_bake_tests` test binary.
- All third-party brought in via CPM `FetchContent`. No git submodules.
- CUDA compiled with `--use_fast_math`, `-lineinfo`, separable compilation off unless needed.
- Compute capability target list: `75;80;86;89;90` (Turing through Hopper). Override via cache var.
- Warnings as errors on host code. Strict warnings on CUDA host portions, relaxed on device.
- A `ninja`-friendly default. No platform-specific build steps in source.

## 6. Core architecture

### 6.1 Cubed sphere grid

Represent the simulation domain as six square 2D grids (the cube faces), each of resolution `N × N`. Total cell count is `6 * N * N`. Faces are indexed 0..5 in the order `+X, -X, +Y, -Y, +Z, -Z`.

Provide:

- `CubedSphereGrid` holding face resolution, total resolution string (for cache keys), and the precomputed neighbor table.
- `face_uv_to_sphere(face, u, v) -> vec3` mapping from local face coords to a unit sphere position, using equiangular cubed-sphere parameterization, not naive normalize.
- `sphere_to_face_uv(p) -> (face, u, v)` for the inverse, used when resampling from other grids.
- `face_uv_to_latlon(face, u, v)` helpers.
- `cell_solid_angle(face, u, v)` for area-weighted averages.

Use equiangular parameterization: face local coords map through `tan(angle)` rather than linear projection. This gives nearly equal-area cells, which matters for erosion and climate stability.

### 6.2 Neighbor table

For each cell on a face boundary, the four-neighbor lookup must cross to an adjacent face. Build a precomputed table at startup keyed by `(face, edge_side)`:

- Adjacent face index.
- Edge mapping: which row or column on the neighbor maps to this edge.
- Coordinate rotation: a sign/swap pair indicating how `(u,v)` on this face's edge maps to `(u,v)` on the neighbor.

Provide:

- `Neighbors4 neighbors(face, i, j)` returning the four neighbor `(face, i, j)` triplets, with no special casing required from the caller.
- A halo exchange function `halo_exchange<T>(Field<T>&)` that fills a 2-cell ghost border around each face from neighboring faces, applying the rotation. All stencil kernels read from ghost borders, never branch on edge.
- Unit tests that walk a ring around each face corner and verify no cell is visited twice.

### 6.3 Planet state

Define `PlanetState` as a struct of `Field<T>` objects. A `Field<T>` is a CUDA device buffer sized `6 * N * N` plus a 2-cell halo per face. Fields:

```
// elevation and crust
Field<float>  bedrock_elevation;
Field<float>  crust_thickness;
Field<float>  crust_density;
Field<float>  crust_age;
Field<uint8>  crust_type;            // continental / oceanic / volcanic / impact-melt

// surface deposits
Field<float>  sediment_thickness;
Field<float>  soil_thickness;
Field<float>  ice_thickness;
Field<float>  water_column;          // zero on dry worlds
Field<float>  vegetation_density;    // zero on dry / lifeless worlds

// climate
Field<float>  surface_temperature_mean;
Field<float>  surface_temperature_range;
Field<float>  precipitation_mean;
Field<float>  atmospheric_moisture;
Field<float2> wind_surface;          // 2D in face-local frame
Field<float>  dust_loading;

// cloud state
Field<float>  cloud_dust_od;
Field<float>  cloud_ice_od;
Field<float>  cloud_water_od;
Field<float>  cloud_top_altitude;

// derived
Field<float4> biome_weights;         // (rock, sand, ice, reserved)
Field<float>  surface_elevation;     // bedrock + sediment + ice + water, recomputed when needed
```

Every field is allocated at construction. Fields not exercised by a given world type stay at zero. Memory cost at N=2048: ~24M cells per scalar, ~100MB per float field. Total state at full allocation: ~3GB. Acceptable.

All fields support save/load to disk via raw binary blobs with a small header (magic, version, dtype, dims, scale, offset). Used by the cache layer and for hand inspection.

### 6.4 Pass system

A pass is a unit of computation that reads some fields and writes others. Interface:

```cpp
class Pass {
public:
    virtual const char* name() const = 0;
    virtual uint64_t version() const = 0;          // bump on algorithm change to invalidate caches
    virtual FieldSet reads() const = 0;
    virtual FieldSet writes() const = 0;
    virtual void declare_params(ParamRegistry&) = 0;
    virtual void run(PlanetState&, const Params&, ProgressSink&) = 0;
};
```

`FieldSet` is a bitset enum over the field roster.

`Params` is a typed key/value map populated from JSON or live UI edits.

`ProgressSink` reports stage, fraction, and elapsed time so the viewer can show progress bars and timing breakdowns.

### 6.5 Pipeline

`Pipeline` owns an ordered list of passes plus the cache layer. Operations:

- `run(state, params)`: execute every pass in order, reading from cache when input hash matches.
- `run_from(state, params, pass_name)`: invalidate cache from `pass_name` onward, re-run from there. Used after parameter changes.
- `set_passes(list)`: swap pass list, used by world-type presets.

Cache key per pass: hash of pass name + pass version + relevant input field content hashes + serialized params. Hash function is xxh3.

Cache storage: `cache/<key>/<field_name>.bin`. Hits load directly to GPU.

### 6.6 Parameter system

`ParamRegistry` is a hierarchical tree of params, each with name, type, default, range, units, tooltip. Types: `int`, `float`, `bool`, `enum<>`, `vec2/vec3`, `gradient`, `string`.

JSON config maps 1:1 with the registry. Hot reload: edits in viewer mutate the registry, the pipeline re-runs affected passes. Save back to JSON on user command.

Presets are JSON files in `presets/` that set a subset of params and a chosen pass list. Loading a preset replaces the entire pass list and resets unaffected params to defaults.

### 6.7 Logging

`Log::info/warn/error` with category tags. A scrollable log panel in the viewer mirrors stdout. Per-pass timings stored alongside the cache, displayed in a timing panel.

## 7. Subsystems

Each is implemented as a `Pass`. Section gives algorithm, fields touched, params exposed, acceptance criteria.

### 7.1 TectonicsPass

**Algorithm.** Simulate plate tectonics on a Goldberg polyhedron (hex grid from subdivided icosahedron, dualized). Plate state: ID, Euler pole, angular velocity. Cell state on hex grid: plate ID, crust thickness, density, age, type. Each timestep advances all plates by their angular velocity, detects boundary cells, classifies each boundary as divergent, convergent (with subtypes), or transform, and applies the corresponding crust changes. Hotspots are stationary in absolute frame and trigger volcanic crust deposition on whatever plate is overhead. Crust thickness produces elevation via Airy isostasy. After the run finishes, resample to cubed sphere using inverse-distance interpolation with a small kernel radius.

**Fields written.** `bedrock_elevation`, `crust_thickness`, `crust_density`, `crust_age`, `crust_type`.

**Fields read.** None (initial pass).

**Params.**
- `tectonics.plate_count` (int, default 12, range 4-32)
- `tectonics.plate_motion_energy` (float, 0-2, default 1)
- `tectonics.sim_duration_myr` (float, default 1500)
- `tectonics.timestep_myr` (float, default 2)
- `tectonics.fragmentation_events` (int, default 2)
- `tectonics.hotspot_count` (int, default 6)
- `tectonics.hotspot_intensity` (float, default 1)
- `tectonics.subduction_uplift_rate` (float, default 0.05 km/Myr)
- `tectonics.continent_continent_uplift_rate` (float, default 0.08 km/Myr)
- `tectonics.divergent_rifting_rate` (float, default 0.04 km/Myr)
- `tectonics.ocean_cooling_coefficient` (float, default 350)
- `tectonics.seed` (uint64)

**Acceptance.** After running with defaults, the cubed-sphere `bedrock_elevation` shows continent-like masses with ridges, basins, and at least two convincing mountain chains. The crust age field has young ridges and old basins. Crater pass disabled, the planet renders recognizably.

### 7.2 ImpactPass

**Algorithm.** Generate a population of craters from a power-law size distribution scaled by a flux parameter. For each crater, pick a random spherical coordinate, an age stamp uniform over a configurable epoch, a diameter sampled from the SFD. Classify morphology by diameter: simple (<3km), complex (3-150km), basin (150-1000km), multi-ring (>1000km). Sort by age, oldest first, then splat each crater analytically into the bedrock heightmap: bowl/central peak/ring structure plus rim plus ejecta blanket out to 2-3 radii. Each crater also bumps `crust_type` to impact-melt in the cavity and adds sediment to the ejecta region. Run as one launch per crater with a per-thread region size, or batch small craters into a single launch.

**Fields written.** `bedrock_elevation`, `sediment_thickness`, `crust_type`.

**Fields read.** `bedrock_elevation`.

**Params.**
- `impacts.flux_multiplier` (float, default 1, range 0-10)
- `impacts.size_distribution_exponent` (float, default 2.0)
- `impacts.min_diameter_km` (float, default 1)
- `impacts.max_diameter_km` (float, default 1500)
- `impacts.epoch_start_myr` (float, default 4000)
- `impacts.epoch_end_myr` (float, default 100)
- `impacts.simple_complex_threshold_km` (float, default 3)
- `impacts.basin_threshold_km` (float, default 150)
- `impacts.multiring_threshold_km` (float, default 1000)
- `impacts.ejecta_extent_radii` (float, default 2.5)
- `impacts.seed` (uint64)

**Acceptance.** Visually plausible crater field. Older craters partially erased by newer ones. Basin-class craters visible at planetary scale. Ejecta blankets visible at small scale in the viewer.

### 7.3 ClimatePass

**Algorithm.** Solve a simplified atmospheric circulation to steady state. Components:

1. Insolation map: top-of-atmosphere flux as a function of latitude, axial tilt, and configurable season averaging.
2. Surface temperature: one-layer energy balance with latitudinal diffusion, elevation lapse rate of 6.5 K/km, plus ice-albedo feedback (ice cells reflect more, cool further).
3. Shallow water solver on the cubed sphere: solve height-and-velocity equations with Coriolis force and damping, forced by horizontal temperature gradients. Use a finite-volume scheme on each face with halo exchange. Stable timestep set by CFL.
4. Moisture transport: an advected scalar with evaporation source (proportional to surface temperature times water column times wind speed; near zero on dry worlds) and precipitation sink (where moist column is supersaturated, especially under orographic uplift).
5. Dust transport: an advected scalar with lift source (where wind shear exceeds threshold and sediment is available) and settling sink (proportional to dust loading).
6. Read out time-averaged fields after convergence: mean wind, mean temperature, precipitation, dust loading.

**Fields written.** `surface_temperature_mean`, `surface_temperature_range`, `precipitation_mean`, `wind_surface`, `atmospheric_moisture`, `dust_loading`.

**Fields read.** `bedrock_elevation`, `sediment_thickness`, `ice_thickness`, `water_column`, `vegetation_density`, `biome_weights` (for albedo).

**Params.**
- `climate.stellar_luminosity_rel` (float, default 1)
- `climate.distance_au` (float, default 1)
- `climate.axial_tilt_deg` (float, default 23.5)
- `climate.rotation_rate_rel` (float, default 1)
- `climate.atmospheric_pressure_bar` (float, default 1)
- `climate.greenhouse_factor` (float, default 1)
- `climate.albedo_rock` (float, default 0.2)
- `climate.albedo_sand` (float, default 0.4)
- `climate.albedo_ice` (float, default 0.7)
- `climate.albedo_vegetation` (float, default 0.15)
- `climate.timestep_hours` (float, default 1)
- `climate.spinup_days` (int, default 60)
- `climate.average_window_days` (int, default 30)
- `climate.dust_lift_threshold` (float)
- `climate.dust_settling_rate` (float)

**Acceptance.** Three-cell circulation visible in the wind field. Temperature gradient from equator to pole. Polar cells cold enough for ice. On a dry world, precipitation near zero everywhere; on a world with positive initial water column, precipitation banded as expected (ITCZ peak, subtropical minimum). No visible cubed-sphere seam artifacts.

### 7.4 ThermalErosionPass

**Algorithm.** For each cell, compute slopes to four neighbors. Where slope exceeds the local angle of repose, transfer sediment to lower neighbors. Angle of repose varies by surface material (loose sand ~30°, weathered rock ~35°, fresh bedrock ~60°+). Iterate to relaxation, typically a few hundred iterations.

**Fields written.** `sediment_thickness`, `bedrock_elevation`.

**Fields read.** `sediment_thickness`, `bedrock_elevation`, `biome_weights`.

**Params.**
- `thermal.iterations` (int, default 400)
- `thermal.angle_repose_sand_deg` (float, default 32)
- `thermal.angle_repose_rock_deg` (float, default 55)
- `thermal.transfer_rate` (float, default 0.3)

**Acceptance.** Sharp impact features rounded slightly. Steep mountain faces show talus aprons. No fields go negative.

### 7.5 AeolianErosionPass

**Algorithm.** Iterative GPU pass. Each iteration:

1. Compute surface wind stress per cell from `wind_surface`.
2. Lift sediment where stress exceeds the lift threshold and sediment reservoir is non-empty. Vegetation reduces lift.
3. Advect lifted material as a transient suspended-sediment scalar using the wind field.
4. Deposit where stress drops below the deposition threshold, especially in basins and lee zones.
5. Abrade bedrock at very high stress where saltation is intense, converting tiny amounts of rock to sediment.

Run for a configurable number of outer iterations until the sediment field reaches dynamic equilibrium. Save snapshots every K iterations so the viewer can scrub through evolution.

**Fields written.** `sediment_thickness`, `bedrock_elevation` (slight), `biome_weights` (indirectly).

**Fields read.** `wind_surface`, `bedrock_elevation`, `sediment_thickness`, `vegetation_density`.

**Params.**
- `aeolian.iterations` (int, default 800)
- `aeolian.lift_threshold` (float)
- `aeolian.deposition_threshold` (float)
- `aeolian.abrasion_rate` (float, default 1e-5)
- `aeolian.vegetation_shielding` (float, default 0.95)
- `aeolian.transport_distance_cells` (float, default 4)

**Acceptance.** Sand accumulates in basins, on lee sides of mountains, in subtropical belts. Streaks visible behind craters. Rock fraction shows where wind has stripped sediment back to bedrock.

### 7.6 GlacialPass

**Algorithm.** Where mean annual temperature is below freezing and net annual precipitation positive, accumulate ice. Ice flows by shallow-ice approximation: velocity proportional to surface slope cubed times thickness squared. Scours bedrock at the ice base proportional to velocity. Calves into water at coasts; sublimates at edges if dry.

**Fields written.** `ice_thickness`, `bedrock_elevation`, `sediment_thickness` (glacial debris).

**Fields read.** `surface_temperature_mean`, `precipitation_mean`, `bedrock_elevation`, `water_column`.

**Params.**
- `glacial.melt_temperature_c` (float, default 0)
- `glacial.flow_coefficient` (float)
- `glacial.scour_rate` (float)
- `glacial.iterations` (int, default 200)

**Acceptance.** Stable polar ice caps. Some glacial scour on high-altitude regions outside polar zones. Caps shrink to nothing if stellar luminosity raised, expand to global if dropped.

### 7.7 HydraulicErosionPass

**Algorithm.** Mei et al. virtual pipes. Each cell has water column, suspended sediment, four pipe outflows. Per iteration: rainfall input from `precipitation_mean`, pipe flux update from elevation differences, water column advection, sediment erosion proportional to velocity capacity, sediment deposition where carrying capacity drops, water evaporation. Run for many iterations.

On dry worlds this pass is excluded from the pipeline by the preset. On wet worlds it runs to equilibrium driven by the climate precipitation field. Provide a "wet era" mode that runs once with a forced rainfall override even on a dry-world preset, leaving carved canyons frozen in place.

**Fields written.** `bedrock_elevation`, `sediment_thickness`, `water_column`.

**Fields read.** `precipitation_mean`, `bedrock_elevation`, `sediment_thickness`, `water_column`, `vegetation_density`.

**Params.**
- `hydraulic.iterations` (int, default 1500)
- `hydraulic.rainfall_multiplier` (float, default 1)
- `hydraulic.wet_era_override_rainfall` (float, default 0)
- `hydraulic.wet_era_duration_iter` (int, default 0)
- `hydraulic.erosion_constant` (float)
- `hydraulic.deposition_constant` (float)
- `hydraulic.evaporation_rate` (float)
- `hydraulic.vegetation_protection` (float, default 0.9)

**Acceptance (when enabled).** Dendritic drainage networks form. Valleys widen downstream. Sediment fans appear at outlets.

### 7.8 WeatheringPass (stub)

Empty implementation that does nothing. Declares `soil_thickness` as written but never writes. Exists so the pipeline graph is complete for future wet/lush builds. Param tree includes a TODO marker. Do not delete; do not implement now.

### 7.9 VegetationGrowthPass (stub)

Empty implementation. Declares `vegetation_density` as written but never writes. Param tree marked TODO. Do not delete; do not implement now.

### 7.10 BiomeClassifierPass

**Algorithm.** Per-cell function from state to a blend of three weights: rock, sand, ice. Rules for the dry-world classifier:

- ice weight from `ice_thickness` saturating to 1 around a few meters.
- sand weight from `sediment_thickness` saturating to 1 over a configurable depth, modulated down by ice presence.
- rock weight is 1 minus the others.

Pluggable interface so wet and lush classifiers can be added later without touching this pass's caller.

**Fields written.** `biome_weights`.

**Fields read.** `sediment_thickness`, `ice_thickness`, `water_column`, `vegetation_density`, `surface_temperature_mean`.

**Params.**
- `biome.sand_saturation_depth_m` (float, default 0.5)
- `biome.ice_saturation_depth_m` (float, default 1)
- `biome.classifier_name` (enum: dry, wet, lush; default dry)

**Acceptance.** Polar regions classified ice. Basins and aeolian-deposit zones classified sand. Rocky highlands classified rock. Smooth transitions, no hard checker artifacts.

### 7.11 BakePass

**Algorithm.** Take final state at working resolution, upsample to output resolutions, encode to KTX2 with per-channel format choices, write EXR sidecars at the working resolution for debug.

Output texture set, all cubemaps:

| Layer | Resolution | Format | Source field(s) |
|---|---|---|---|
| elevation | 16384 | BC6H (R) | `bedrock_elevation + sediment_thickness + ice_thickness` |
| normal | 16384 | BC5 (RG) | derived from elevation |
| sediment | 16384 | BC4 (R) | `sediment_thickness` |
| ice | 8192 | BC4 (R) | `ice_thickness` |
| water | 8192 | BC4 (R) | `water_column` |
| lithology | 8192 | BC7 (RGBA) | `crust_type` + `crust_age` |
| biome_weights | 8192 | BC7 (RGBA) | `biome_weights` |
| temperature | 4096 | BC6H (RG) | mean and range |
| precipitation_dust | 4096 | BC6H (RG) | precipitation and dust |
| wind | 4096 | BC6H (RG) | `wind_surface` |
| clouds | 4096 | BC6H (RGBA) | cloud ODs and top altitude |

A `manifest.json` next to the textures describes each layer: filename, channels, encoding, value scale, value offset, projection (cubed sphere equiangular), face order. The renderer reads this manifest. EXR sidecars dumped to `out/debug/*.exr` at working resolution for hand inspection.

Upsampling uses bicubic for scalar fields, plus optional procedural detail noise modulated by biome for elevation specifically (controlled by a param, off by default; the renderer is expected to add its own fine noise at tile streaming time).

**Params.**
- `bake.elevation_resolution` (int, default 16384)
- `bake.medium_resolution` (int, default 8192)
- `bake.climate_resolution` (int, default 4096)
- `bake.output_directory` (string, default `./out`)
- `bake.write_exr_debug` (bool, default true)
- `bake.detail_noise_amplitude` (float, default 0)

**Acceptance.** Output directory contains 11 KTX2 files, a manifest, and matching EXR debug files. KTX2 files validate with `ktxinfo`. Re-running bake without changes produces byte-identical files.

## 8. Debug viewer

The viewer is the primary tuning interface. Built with GLFW, OpenGL 4.5, ImGui.

### 8.1 Layout

- Central viewport: 3D sphere rendered with OpenGL, textured from the currently selected field. CUDA-OpenGL interop maps the field buffer into a GL texture each frame (or once per pass completion).
- Right dock: parameter tree. Tree mirrors `ParamRegistry`. Edit any value, the pass becomes dirty and downstream caches invalidate. A "Run" button re-runs the dirty subset.
- Left dock: pipeline view. Pass list with status badges (clean, dirty, running, cached). Click a pass to show its inputs/outputs and last timing. Right-click to re-run from this pass.
- Bottom dock: log panel, timing panel, histogram panel for the currently selected field.
- Top bar: world preset selector, resolution selector (preview/dev/production), save preset, load preset, "Bake" button (disabled until pipeline is clean).

### 8.2 Sphere rendering

A unit-sphere mesh (icosphere, subdivision level 6 enough). Vertex shader applies camera. Fragment shader samples the cubed-sphere texture: for the fragment's world-space normal, compute which cube face it belongs to, the local face uv, sample the texture, apply the selected colormap.

The texture is a CUDA-mapped resource: each pass writes to its CUDA buffer, the viewer rebinds the buffer as a GL texture (or uses CUDA arrays + cudaGraphicsMapResources). One texture per active field; only the selected one is bound.

Camera: orbit around origin with pitch/yaw/zoom. Reset button. Lock to lat/lon optional.

Hover: cast ray to sphere, compute (face, u, v), display lat/lon, all field values at that cell.

### 8.3 Field view modes

Per field, a default colormap and value scaling:

- elevation: hypsometric, blue below zero, green-brown above, white peaks. Configurable sea level even on dry worlds (visualization only).
- temperature: blue to red, 250K to 320K.
- precipitation: white to deep blue, 0 to 5 mm/day.
- wind: line integral convolution overlay on a base colormap, or 2D arrow glyphs.
- biome weights: RGB direct, palette swappable.
- crust age: red to blue, 0 to 4 Gyr.
- crater age: hot to cold colormap.
- Anything scalar: viridis fallback.

Colormaps in `render/colormap.h` as device-side LUTs.

### 8.4 Time scrubbing

The TectonicsPass can checkpoint its state every K Myr. The viewer can scrub a timeline and view the planet at any checkpoint. Same applies to AeolianErosionPass and HydraulicErosionPass iteration snapshots. Backed by the on-disk cache.

### 8.5 Pre-bake review

A "Bake" workflow that opens a modal:

- Thumbnail of each output layer at preview resolution.
- Histograms.
- Numeric sanity checks: elevation range, biome fractions, ice coverage percent, mean temperature, mean precipitation.
- A list of warnings if any check fails (e.g. >70% ice, NaN cells, biome weights not summing to 1).
- "Approve & Bake" button. Cancels return to viewer. On approval, run BakePass to full resolution and write outputs.

## 9. Workflow

Typical session:

1. Launch `planet_bake`, optionally with `--preset dry_rocky --seed 42`.
2. Viewer opens, runs the pipeline at preview resolution (N=512 per face). First run ~15-60 seconds.
3. Inspect elevation, temperature, wind, biome on the globe.
4. Tweak params, e.g. shift hotspot count, axial tilt. Re-run the dirty subset.
5. Step up to dev resolution (N=1024) for a closer look. Iterate.
6. When satisfied, step to production resolution (N=4096) and run full pipeline (~5-20 min depending on GPU).
7. Open the bake review modal, verify checks pass.
8. Approve. Bake writes textures and manifest to `out/`.
9. Renderer (separate process) loads from `out/`.

CLI mode: `planet_bake --config <file> --no-viewer --bake` runs end to end without UI, exits when bake completes. For batch experiments.

## 10. Implementation milestones

Build in this order. Each milestone produces a runnable binary. Do not skip ahead.

### M0. Skeleton (1-2 days)

- Folder, CMakeLists, CPM, third-party fetch.
- `main.cpp` opens a GLFW window with ImGui and renders an empty viewport.
- Hello-world CUDA kernel runs at startup, prints device info.
- Test target builds and runs.

Acceptance: `cmake --build` succeeds, binary opens a window, exits cleanly.

### M1. Cubed sphere + neighbor table (2-3 days)

- Implement `CubedSphereGrid`, `Field<T>`, `NeighborTable`, halo exchange.
- Unit tests for neighbor topology and halo correctness.

Acceptance: `planet_bake_tests` passes. A debug kernel can write `latitude` into a field; the value matches between neighboring cells across face seams.

### M2. Planet state + debug viewer baseline (3-5 days)

- Implement `PlanetState` with all fields allocated.
- OpenGL sphere mesh, CUDA-OpenGL interop, fragment shader that samples a cubed-sphere texture via face math.
- Field selector dropdown, colormap selector, orbit camera, hover info.
- A dummy "noise" pass that fills `bedrock_elevation` with a deterministic 3D noise so there is something to look at.

Acceptance: Viewer renders the noisy planet, hovering shows lat/lon and elevation, all six faces are correctly oriented, no visible seams in the texture sampling.

### M3. Pass and pipeline infrastructure (2-3 days)

- `Pass` interface, `Pipeline`, `ParamRegistry`, JSON config I/O, cache.
- Convert the dummy noise pass into a proper `Pass`.
- Pipeline view in viewer with pass list, status badges, run button.

Acceptance: Tweaking the noise frequency in the UI dirties the pass, re-running takes effect. Restarting the app re-reads the cache and skips clean passes.

### M4. TectonicsPass (5-7 days)

- Goldberg polyhedron mesh generator.
- Hex-grid plate sim: plates, Euler poles, boundary classification, crust updates, hotspots, isostasy.
- Resample hex grid to cubed sphere.
- Checkpoints every K Myr saved to cache for time-scrubbing.

Acceptance: A 1.5-Gyr run produces a planet with continents, mountain chains, mid-ocean ridges (visible in age field), and 6 hotspot tracks. Re-seeding produces a different but qualitatively similar planet.

### M5. ImpactPass (2-3 days)

- SFD sampler, crater morphology, ejecta, age-ordered splatting.

Acceptance: Crater field visible, basin-class craters present, older craters partially erased by newer ones.

### M6. ClimatePass (7-10 days, hardest piece)

- Shallow water solver on cubed sphere, with halo-exchange Coriolis force, finite-volume scheme.
- Moisture and dust transport as advected scalars.
- Energy balance for surface temperature.
- Steady-state convergence loop.

Acceptance: Wind field shows three-cell structure. Temperature smooth and physical. Dust loading non-uniform, accumulates in dry basins. Re-running with axial tilt 0 produces a banded climate without seasons; axial tilt 30 produces strong polar variability.

### M7. ThermalErosionPass (1-2 days)

Acceptance: Steep impact rims relaxed slightly, sand piles repose correctly.

### M8. AeolianErosionPass (3-5 days)

- Lift / advect / deposit kernels driven by wind.
- Iteration snapshots cached for scrubbing.

Acceptance: Sand seas form in basins. Streaks behind craters. Subtropical sand bands visible.

### M9. GlacialPass (2-3 days)

Acceptance: Polar caps form. Caps respond correctly to luminosity changes.

### M10. HydraulicErosionPass, configured for "wet era" override (3-5 days)

Implement the pass, but in the dry preset it is included with iterations=0 by default so it does nothing. Param `wet_era_duration_iter` can be set non-zero to carve canyons.

Acceptance: With override at 500 iterations, valley networks carve into highlands. With override at 0, nothing happens.

### M11. WeatheringPass and VegetationGrowthPass stubs (1 hour)

Empty `Pass` implementations registered in the pipeline registry. Do not include in dry preset.

### M12. BiomeClassifierPass (1 day)

Dry classifier implemented. `classifier_name` enum has wet and lush as "not implemented" placeholders.

### M13. BakePass + KTX2/EXR I/O (4-6 days)

- Upsample kernels (bicubic).
- BC encoder integration. Recommend NVIDIA NVTT 3 (CUDA-accelerated) or compute-shader BCn encoder; if unavailable, use a CPU encoder like `bc7enc_rdo` initially and replace later.
- KTX2 writer wiring.
- Manifest writer.
- Pre-bake review modal in viewer.

Acceptance: `out/` populated with the texture set. Re-bake byte-identical. Manifest validates.

### M14. Presets and tuning polish (2-3 days)

- `dry_rocky.json` tuned to produce a visually compelling default.
- `wet_rocky.json` and `lush.json` as placeholders with TODO notes.
- Per-param tooltips and units.
- "Reset to default" buttons.

Acceptance: A user with no prior context can launch, load `dry_rocky`, and produce a credible texture set in under five minutes without touching parameters.

### M15. CLI mode + final cleanup (1-2 days)

- `--no-viewer --bake` headless mode.
- Doctor command `--check` that validates the build.
- README rewritten with usage examples.

Acceptance: Headless bake from a JSON config produces identical output to viewer-driven bake.

Total estimated time: roughly 40-60 days for one engineer, less with Claude Code's assistance.

## 11. Out of scope

Do not implement, do not stub elaborate placeholders, do not architect for:

- Multi-GPU.
- Distributed simulation across machines.
- Direct path tracer integration. The contract is files on disk only.
- Real-time planet rendering inside this tool beyond the simple textured-sphere debug view. No path tracing here.
- Tile-based local detail textures. The path tracer's streaming layer handles that.
- Vulkan or DX12 paths. CUDA only on the compute side.
- Asset import (heightmaps from other sources).
- Networking, telemetry, crash reporting.
- Localization.
- A full physics-based GCM. The shallow-water + moisture approach is the ceiling.

## 12. Extension notes for wet and lush worlds

These are not implemented in this milestone, but the architecture must support them with no rewrites:

- `water_column`, `soil_thickness`, `vegetation_density` are already allocated and zero on dry worlds. Wet/lush presets initialize them differently and enable the relevant passes.
- `ClimatePass` already evaluates evaporation source from `water_column`. Setting water nonzero in basins activates the wet-world moisture cycle without code changes.
- `HydraulicErosionPass` already exists. Wet preset includes it with continuous iterations instead of wet-era override.
- `WeatheringPass` and `VegetationGrowthPass` are stubbed. Lush preset will register them with implementations.
- `BiomeClassifierPass` `classifier_name` enum already has wet and lush slots. Implement those functions when adding the world types.
- Output texture set already includes water, soil, and vegetation layers. They are zero on dry worlds. Renderer code that handles them can be added without changing the manifest schema.

## 13. Naming, style, conventions

- C++: `snake_case` for variables and functions, `PascalCase` for types, `UPPER_SNAKE` for constants. Files lowercase with underscores.
- CUDA: device functions prefixed `d_`, kernels suffixed `_kernel`. One kernel per logical step, no megakernels.
- Headers self-contained, every `.h` has its own include guard via `#pragma once`.
- No raw `new`/`delete`. Use `std::unique_ptr` and the GPU memory wrapper in `gpu/memory.h`.
- All CUDA calls wrapped with `CUDA_CHECK(...)` from `gpu/cuda_check.h`.
- Comments follow project comment style: no space after `//`, one-line max, structured section headers with `//====================================`. No em dashes anywhere, use commas.
- Logging via `Log::` macros, never `printf` or `cout` in non-debug paths.
- Determinism: every random source seeded explicitly. No `std::random_device` in hot paths.

## 14. Definition of done for the whole project

The project is done when:

1. From a clean checkout, `cmake --build` produces `planet_bake` and `planet_bake_tests` on Windows and Linux.
2. `planet_bake_tests` passes.
3. `planet_bake --preset dry_rocky` opens a viewer that shows a credible rocky planet within 60 seconds at preview resolution.
4. Approving a bake at production resolution writes a complete texture set to `out/` along with `manifest.json`.
5. The renderer team can load `out/manifest.json` and the referenced KTX2 files using only the documented schema. They never need to read `planet_bake` source.
6. The PLAN.md extension notes for wet/lush remain accurate: turning on the relevant passes and presets would not require core refactors.
