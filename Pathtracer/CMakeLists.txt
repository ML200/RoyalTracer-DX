cmake_minimum_required(VERSION 3.25.2)
set(CMAKE_SYSTEM_VERSION "10.0.22621.0")
project(Pathtracer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA as a first-class language so `.cu` files compile natively
# and tiny-cuda-nn's CMake can configure itself. These two overrides are
# no-ops when CUDA is new enough for the installed MSVC (CUDA 12.9+ for
# VS 17.14), but kept so older-toolkit builds still compile:
#   1. nvcc's own host_config.h check (-allow-unsupported-compiler)
#   2. MSVC STL's yvals_core.h static_assert (_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -allow-unsupported-compiler -D_ALLOW_COMPILER_AND_STL_VERSION_MISMATCH")
# Let NVCC parallelize its own internal passes per .cu file. Requires CUDA 11.2+.
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--threads=0>)
# Compile for the installed GPU only. tcnn picks this up before its autodetect
# runs, so Pathtracer and tcnn target the same architecture (avoids an arch
# mismatch in tcnn's TCNN_MIN_GPU_ARCH static_assert).
set(CMAKE_CUDA_ARCHITECTURES native)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ───────────────────────── DirectX Agility SDK ─────────────────────────────
set(D3D12_AGILITY_ROOT "${CMAKE_SOURCE_DIR}/microsoft.direct3D.d3d12.1.719.0-preview/build/native")
set(D3D12_AGILITY_INCLUDE "${D3D12_AGILITY_ROOT}/include")
set(D3D12_AGILITY_LIB "${D3D12_AGILITY_ROOT}/bin/x64")

# ───────────────────────────── Streamline paths ──────────────────────────────
set(SL_ROOT        "${CMAKE_SOURCE_DIR}/streamline")
set(SL_INCLUDE_DIR "${SL_ROOT}/include")
set(SL_LIB_PATH    "${SL_ROOT}/lib/x64/sl.interposer.lib")
set(SL_BIN_DIR     "${SL_ROOT}/bin/x64")

# ───────────────────────── NVIDIA OMM SDK ───────────────────────────────────
set(OMM_SDK_ROOT "${CMAKE_SOURCE_DIR}/third_party/omm-sdk")

# ───────────────────────── ONNX Runtime + DirectML ──────────────────────────
set(ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-directml/ort")

set(ORT_INCLUDE_DIR "${ORT_ROOT}/build/native/include")
set(ORT_BIN_DIR     "${ORT_ROOT}/runtimes/win-x64/native")
set(ORT_LIB_DIR     "${ORT_BIN_DIR}")

# Hard checks so you fail fast if the paths are wrong
if(NOT EXISTS "${ORT_INCLUDE_DIR}/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "ORT headers not found: ${ORT_INCLUDE_DIR} (expected onnxruntime_cxx_api.h)")
endif()

if(NOT EXISTS "${ORT_LIB_DIR}/onnxruntime.lib")
    message(FATAL_ERROR "ORT import lib not found: ${ORT_LIB_DIR}/onnxruntime.lib")
endif()

if(NOT EXISTS "${ORT_BIN_DIR}/onnxruntime.dll")
    message(FATAL_ERROR "ORT runtime DLL not found: ${ORT_BIN_DIR}/onnxruntime.dll")
endif()

message(STATUS "ORT include : ${ORT_INCLUDE_DIR}")
message(STATUS "ORT lib dir : ${ORT_LIB_DIR}")
message(STATUS "ORT bin dir : ${ORT_BIN_DIR}")

# ───────────────────────── CUDA Toolkit (CUDA/D3D12 interop) ─────────────────
find_package(CUDAToolkit REQUIRED)
message(STATUS "CUDA Toolkit : ${CUDAToolkit_VERSION}")

# ───────────────────────── tiny-cuda-nn (vendored submodule) ─────────────────
# Build tcnn as a static lib, no examples/benchmarks/tests.
set(TCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_EXAMPLES  OFF CACHE BOOL "" FORCE)
set(TCNN_BUILD_TESTS     OFF CACHE BOOL "" FORCE)
# Skip offline compilation of every fused fwd/bwd kernel permutation — tcnn
# instead JIT-compiles the single network config we actually instantiate via
# tcnn::create_from_config at startup (cached on disk). Cuts tcnn build time
# dramatically. Requires TCNN_BUILD_WITH_RTC (ON by default).
set(TCNN_BUILD_NO_FWD_BWD ON  CACHE BOOL "" FORCE)
set(TCNN_BUILD_WITH_RTC   ON  CACHE BOOL "" FORCE)
add_subdirectory(third_party/tiny-cuda-nn)


# ────────────────────────── extra assets to copy ─────────────────────────────
set(INCLUDES_DIR   "${CMAKE_SOURCE_DIR}/include")
set(SHADERS_DIR   "${CMAKE_SOURCE_DIR}/shaders")


# ─────────────────────── Sky star map (NASA SVS 4851) ──────────────────────
# Equirectangular HDR star + Milky Way map sampled by EvaluateStars
# (SunSampler_v8.hlsli). 8K variant is ~120 MB so it's gitignored and
# downloaded on demand. The runtime loader handles a missing file gracefully
# (sky falls back to night base), so a failed download isn't fatal — just
# means no stars until the next configure pulls it down.
#
# Override the URL at configure time if NASA reorganises or you want a
# different resolution / mirror:
#   cmake -DSKY_STARS_URL=https://my.mirror/sky_stars.exr ...
set(SKY_STARS_URL
    "https://svs.gsfc.nasa.gov/vis/a000000/a004800/a004851/starmap_2020_8k.exr"
    CACHE STRING "URL to fetch Pathtracer/include/sky_stars.exr from if missing")
set(SKY_STARS_FILE "${INCLUDES_DIR}/sky_stars.exr")

if(NOT EXISTS "${SKY_STARS_FILE}" AND SKY_STARS_URL)
    message(STATUS "[SkyStars] ${SKY_STARS_FILE} not found")
    message(STATUS "[SkyStars] Downloading from ${SKY_STARS_URL} (~120 MB, one time)")
    file(DOWNLOAD
        "${SKY_STARS_URL}"
        "${SKY_STARS_FILE}"
        SHOW_PROGRESS
        TIMEOUT 600
        TLS_VERIFY ON
        STATUS _skystars_status)
    list(GET _skystars_status 0 _skystars_code)
    if(NOT _skystars_code EQUAL 0)
        list(GET _skystars_status 1 _skystars_msg)
        message(WARNING
            "[SkyStars] Download failed (code ${_skystars_code}): ${_skystars_msg}\n"
            "  Build will continue; sky will use night base only at runtime.\n"
            "  Set -DSKY_STARS_URL=<url> at configure if the default is broken,\n"
            "  or drop the file into Pathtracer/include/sky_stars.exr manually.")
        file(REMOVE "${SKY_STARS_FILE}")
    else()
        file(SIZE "${SKY_STARS_FILE}" _skystars_size)
        message(STATUS "[SkyStars] Downloaded ${_skystars_size} bytes")
    endif()
endif()


# ──────────────── Planet-scale cloud coverage (NASA BMNG) ──────────────────
# Equirectangular 8192×4096 TIFF from NASA's "Blue Marble: Clouds" composite
# (Visible Earth ID 57747). Sampled by the cloud integrator (Clouds_v8.hlsli
# → g_cloudCoverage at register t43) to gate the procedural noise body with
# real-world cloud climatology — gives proper continental-scale weather
# patterns instead of uniform global coverage. ~34 MB, one-time download.
#
# Override at configure time if NASA reorganises or a mirror is preferred:
#   cmake -DCLOUD_COVERAGE_URL=https://my.mirror/cloud_combined_8192.tif ...
# Setting URL to empty disables the download entirely and the runtime
# falls back to procedural-only coverage.
set(CLOUD_COVERAGE_URL
    "https://eoimages.gsfc.nasa.gov/images/imagerecords/57000/57747/cloud_combined_8192.tif"
    CACHE STRING "URL to fetch Pathtracer/include/cloud_coverage.tif from if missing")
set(CLOUD_COVERAGE_FILE "${INCLUDES_DIR}/cloud_coverage.tif")

if(NOT EXISTS "${CLOUD_COVERAGE_FILE}" AND CLOUD_COVERAGE_URL)
    message(STATUS "[CloudCoverage] ${CLOUD_COVERAGE_FILE} not found")
    message(STATUS "[CloudCoverage] Downloading from ${CLOUD_COVERAGE_URL} (~34 MB, one time)")
    file(DOWNLOAD
        "${CLOUD_COVERAGE_URL}"
        "${CLOUD_COVERAGE_FILE}"
        SHOW_PROGRESS
        TIMEOUT 600
        TLS_VERIFY ON
        STATUS _cloudcov_status)
    list(GET _cloudcov_status 0 _cloudcov_code)
    if(NOT _cloudcov_code EQUAL 0)
        list(GET _cloudcov_status 1 _cloudcov_msg)
        message(WARNING
            "[CloudCoverage] Download failed (code ${_cloudcov_code}): ${_cloudcov_msg}\n"
            "  Build will continue; clouds will use procedural-only coverage at runtime.\n"
            "  Set -DCLOUD_COVERAGE_URL=<url> at configure if the default is broken,\n"
            "  or drop the file into Pathtracer/include/cloud_coverage.tif manually.")
        file(REMOVE "${CLOUD_COVERAGE_FILE}")
    else()
        file(SIZE "${CLOUD_COVERAGE_FILE}" _cloudcov_size)
        message(STATUS "[CloudCoverage] Downloaded ${_cloudcov_size} bytes (8192×4096 TIFF)")
    endif()
endif()


# ────────────────────────── DirectXTex ─────────────────────────────
file(GLOB DIRECTXTEX_SOURCES "DirectXTex/*.cpp")
add_library(DirectXTex STATIC ${DIRECTXTEX_SOURCES})
target_include_directories(DirectXTex PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/DirectXTex
        ${D3D12_AGILITY_INCLUDE}
)
target_compile_features(DirectXTex PUBLIC cxx_std_17)
target_link_libraries(DirectXTex PUBLIC dxguid ole32)


# ───────────────────────── source files ──────────────────────────────────────
add_executable(Pathtracer WIN32
        rdn/Main.cpp
        rdn/nv_helpers_dx12/BottomLevelASGenerator.cpp
        rdn/nv_helpers_dx12/RaytracingPipelineGenerator.cpp
        rdn/nv_helpers_dx12/RootSignatureGenerator.cpp
        rdn/nv_helpers_dx12/ShaderBindingTableGenerator.cpp
        rdn/nv_helpers_dx12/TopLevelASGenerator.cpp
        rdn/Renderer.cpp
        rdn/DXSample.cpp
        rdn/manipulator.cpp
        rdn/stdafx.cpp
        rdn/Win32Application.cpp
        rdn/nv_helpers_dx12/BottomLevelASGenerator.h
        rdn/nv_helpers_dx12/RaytracingPipelineGenerator.h
        rdn/nv_helpers_dx12/RootSignatureGenerator.h
        rdn/nv_helpers_dx12/ShaderBindingTableGenerator.h
        rdn/nv_helpers_dx12/TopLevelASGenerator.h
        rdn/Renderer.h
        rdn/DXSample.h
        rdn/manipulator.h
        rdn/stdafx.h
        rdn/Win32Application.h
        rdn/LightTree.h
        src/Components/Vertex.h
        src/Util/ObjLoader.h
        rdn/Diagnostics.h
        rdn/LightTree.h
        src/Util/stb_image.h
        src/Util/stb_image_resize2.h
        rdn/CameraRecorder.cpp
        rdn/CameraRecorder.h
        rdn/CameraPathSimulator.cpp
        rdn/CameraPathSimulator.h
        rdn/ResourceStateTracker.h

        rdn/Core/DeviceContext.h
        rdn/Core/DeviceContext.cpp
        rdn/Core/ResourceFactory.h
        rdn/Interop/CudaInterop.h
        rdn/Interop/CudaInterop.cpp
        rdn/NRC/NrcNetwork.h
        rdn/NRC/NrcNetwork.cu
        rdn/Camera/Camera.h
        rdn/Camera/Camera.cpp
        rdn/Editor/Editor.h
        rdn/Editor/Editor.cpp
        rdn/PostProcess/DLSSManager.h
        rdn/PostProcess/DLSSManager.cpp
        rdn/Raytracing/PassSystem.h
        rdn/Raytracing/PassSystem.cpp
        rdn/Scene/Scene.h
        rdn/Scene/Scene.cpp
        rdn/Scene/AssetLoader.h
        rdn/Scene/AssetLoader.cpp
        rdn/Scene/OmmBuilder.h
        rdn/Scene/OmmBuilder.cpp
        rdn/Renderer_Pipeline.cpp
        rdn/Common.h

        rdn/ThirdPartyImpl.cpp
        rdn/Lighting/LightTreeRefit.h

        # Planet streaming (Phases 1-4) - all planet sources except planet_tests.cpp
        rdn/planet/coordinate_system.h
        rdn/planet/coordinate_system.cpp
        rdn/planet/cube_sphere.h
        rdn/planet/cube_sphere.cpp
        rdn/planet/restricted_quadtree.h
        rdn/planet/restricted_quadtree.cpp
        rdn/planet/blas_cells.h
        rdn/planet/blas_cells.cpp
        rdn/planet/chunk_mesh.h
        rdn/planet/blas_pool.h
        rdn/planet/blas_pool.cpp
        rdn/planet/scratch_pool.h
        rdn/planet/scratch_pool.cpp
        rdn/planet/upload_ring.h
        rdn/planet/upload_ring.cpp
        rdn/planet/heightmap_source.h
        rdn/planet/heightmap_source.cpp
        rdn/planet/heightmap_procedural.h
        rdn/planet/heightmap_procedural.cpp
        rdn/planet/heightmap_cubemap.h
        rdn/planet/heightmap_cubemap.cpp
        rdn/planet/worker_pool.h
        rdn/planet/worker_pool.cpp
        rdn/planet/tessellator.h
        rdn/planet/tessellator.cpp
        rdn/planet/queue_sync.h
        rdn/planet/queue_sync.cpp
        rdn/planet/geometry_pool.h
        rdn/planet/geometry_pool.cpp
        rdn/planet/generation.h
        rdn/planet/generation.cpp
        rdn/planet/rock_scatter.h
        rdn/planet/rock_scatter.cpp
        rdn/planet/stream_orchestrator.h
        rdn/planet/stream_orchestrator.cpp
        rdn/planet/tlas_builder.h
        rdn/planet/tlas_builder.cpp

        # Engine layer
        engine/Core/Transform.h
        engine/Input/InputManager.h
        engine/Input/InputManager.cpp
        engine/Scene/GameObject.h
        engine/Scene/SceneDefinition.h
        engine/Scene/SceneManager.h
        engine/Scene/SceneManager.cpp
        engine/Scene/EmissiveCubes.h
        engine/Scene/EmissiveCubes.cpp
        engine/Camera/FlyCamController.h
        engine/Camera/FlyCamController.cpp
        engine/EngineApp.h
        engine/EngineApp.cpp

        # ImGui
        lib/imgui/imgui.cpp
        lib/imgui/imgui_draw.cpp
        lib/imgui/imgui_tables.cpp
        lib/imgui/imgui_widgets.cpp
        lib/imgui/imgui_impl_dx12.cpp
        lib/imgui/imgui_impl_win32.cpp
)

# ───────────────────────── include directories ───────────────────────────────
target_include_directories(Pathtracer PRIVATE
        ${D3D12_AGILITY_INCLUDE}
        ${SL_INCLUDE_DIR}
        ${ORT_INCLUDE_DIR}
        ${OMM_SDK_ROOT}/include
        ${CMAKE_SOURCE_DIR}/lib/imgui)

# NOMINMAX kills the windows.h min/max macros project-wide so std::min,
# std::max, std::numeric_limits<T>::max() etc. compile in any TU that
# transitively pulls in windows.h (d3d12.h, dxgi1_4.h, ...). Required since
# adding tinyexr and any other modern C++ library that uses std::max().
target_compile_definitions(Pathtracer PRIVATE NOMINMAX)

# ──────────────── Streamline import‐lib as an IMPORTED static library ────────
add_library(sl_interposer STATIC IMPORTED GLOBAL)
set_target_properties(sl_interposer PROPERTIES
        IMPORTED_LOCATION           "${SL_LIB_PATH}"
        IMPORTED_LOCATION_DEBUG     "${SL_LIB_PATH}"
        IMPORTED_LOCATION_RELEASE   "${SL_LIB_PATH}"
)
# ────────────── ONNX Runtime imported library (DLL + import lib) ─────────────
add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
        IMPORTED_IMPLIB   "${ORT_LIB_DIR}/onnxruntime.lib"
        IMPORTED_LOCATION "${ORT_BIN_DIR}/onnxruntime.dll"
)

# ────────────── NVIDIA OMM SDK imported library + deps ───────────────────────
add_library(omm_sdk STATIC IMPORTED GLOBAL)
set_target_properties(omm_sdk PROPERTIES
        IMPORTED_LOCATION "${OMM_SDK_ROOT}/lib/omm-lib.lib"
)
add_library(omm_lz4 STATIC IMPORTED GLOBAL)
set_target_properties(omm_lz4 PROPERTIES
        IMPORTED_LOCATION "${OMM_SDK_ROOT}/lib/lz4.lib"
)
add_library(omm_xxhash STATIC IMPORTED GLOBAL)
set_target_properties(omm_xxhash PROPERTIES
        IMPORTED_LOCATION "${OMM_SDK_ROOT}/lib/xxhash.lib"
)


# ───────────────────────── link libraries ────────────────────────────────────
target_link_libraries(Pathtracer PRIVATE
        DirectXTex
        sl_interposer
        onnxruntime
        omm_sdk omm_lz4 omm_xxhash
        CUDA::cudart_static
        tiny-cuda-nn
        d3d12.lib
        dxgi.lib dxguid.lib
        dxcompiler.lib d3dcompiler.lib windowsapp)

target_link_options(Pathtracer PRIVATE /FORCE:MULTIPLE)


# ────────────────── copy the contents of 'includes' on every build ─────────
add_custom_command(
        TARGET Pathtracer
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${INCLUDES_DIR}"
        "$<TARGET_FILE_DIR:Pathtracer>"
        COMMENT "Copying contents of 'includes' into build output directory."
)

# ────────────────── copy the contents of 'shaders' on every build ─────────
add_custom_command(
        TARGET Pathtracer
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SHADERS_DIR}"
        "$<TARGET_FILE_DIR:Pathtracer>"
        COMMENT "Copying contents of 'shaders' into build output directory."
)

message(STATUS "Streamline include: ${SL_INCLUDE_DIR}")
message(STATUS "Streamline lib    : ${SL_LIB_PATH}")
message(STATUS "Streamline bin    : ${SL_BIN_DIR}")
message(STATUS "Includes dir      : ${INCLUDES_DIR}")

# ────────────── copy‐assets target ──────────────
add_custom_target(copy_assets ALL
        COMMENT "Copying include/ and shaders/ to runtime directory"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${INCLUDES_DIR}"
        "$<TARGET_FILE_DIR:Pathtracer>"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SHADERS_DIR}"
        "$<TARGET_FILE_DIR:Pathtracer>"
)

add_dependencies(Pathtracer copy_assets)

add_custom_command(TARGET Pathtracer POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${D3D12_AGILITY_LIB}/D3D12Core.dll"
        "${D3D12_AGILITY_LIB}/d3d12SDKLayers.dll"
        "${D3D12_AGILITY_LIB}/D3D12StateObjectCompiler.dll"
        "$<TARGET_FILE_DIR:Pathtracer>"
        COMMENT "Copying Agility D3D12Core.dll to output folder."
)

# ───────────────────────── copy ORT + DirectML DLLs ─────────────────────────
set(ORT_DLLS
        "${ORT_BIN_DIR}/onnxruntime.dll"
        "${ORT_BIN_DIR}/onnxruntime_providers_shared.dll"
        "${ORT_BIN_DIR}/onnxruntime_providers_directml.dll"
        "${ORT_BIN_DIR}/DirectML.dll"
)

foreach(dll_path IN LISTS ORT_DLLS)
    if(EXISTS "${dll_path}")
        add_custom_command(TARGET Pathtracer POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll_path}"
                "$<TARGET_FILE_DIR:Pathtracer>"
                COMMENT "Copying ORT runtime: ${dll_path}"
        )
    else()
        message(STATUS "ORT optional DLL missing (ok): ${dll_path}")
    endif()
endforeach()

# ───────────────────────── Planet streaming module (Phase 1) ────────────────
# CPU chunk system: cube-sphere quadtree, FP64 coordinates, visibility/LOD pass.
# Standalone test executable - no D3D12 / CUDA, links only the C++ runtime.
# The planet/*.cpp sources join the Pathtracer target at Phase 5 integration.
add_executable(PlanetTests
        rdn/planet/coordinate_system.h
        rdn/planet/coordinate_system.cpp
        rdn/planet/cube_sphere.h
        rdn/planet/cube_sphere.cpp
        rdn/planet/restricted_quadtree.h
        rdn/planet/restricted_quadtree.cpp
        rdn/planet/blas_cells.h
        rdn/planet/blas_cells.cpp
        rdn/planet/chunk_mesh.h
        rdn/planet/heightmap_source.h
        rdn/planet/heightmap_source.cpp
        rdn/planet/heightmap_procedural.h
        rdn/planet/heightmap_procedural.cpp
        rdn/planet/worker_pool.h
        rdn/planet/worker_pool.cpp
        rdn/planet/tessellator.h
        rdn/planet/tessellator.cpp
        rdn/planet/planet_tests.cpp
)
target_compile_features(PlanetTests PRIVATE cxx_std_23)
