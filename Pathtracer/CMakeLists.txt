cmake_minimum_required(VERSION 3.25.2)
set(CMAKE_SYSTEM_VERSION "10.0.22621.0")
project(Pathtracer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(D3D12_AGILITY_ROOT "${CMAKE_SOURCE_DIR}/microsoft.direct3D.d3d12.1.719.0-preview/build/native")
set(D3D12_AGILITY_INCLUDE "${D3D12_AGILITY_ROOT}/include")
set(D3D12_AGILITY_LIB "${D3D12_AGILITY_ROOT}/bin/x64")

set(SL_ROOT "${CMAKE_SOURCE_DIR}/streamline" CACHE PATH "NVIDIA Streamline SDK root")
set(SL_INCLUDE_DIR "${SL_ROOT}/include")
set(SL_LIB_PATH    "${SL_ROOT}/lib/x64/sl.interposer.lib")
set(SL_BIN_DIR     "${SL_ROOT}/bin/x64")

set(OMM_SDK_ROOT "${CMAKE_SOURCE_DIR}/third_party/omm-sdk")

set(INCLUDES_DIR   "${CMAKE_SOURCE_DIR}/include")
set(SHADERS_DIR   "${CMAKE_SOURCE_DIR}/shaders")

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

file(GLOB DIRECTXTEX_SOURCES "DirectXTex/*.cpp")
add_library(DirectXTex STATIC ${DIRECTXTEX_SOURCES})
target_include_directories(DirectXTex PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/DirectXTex
        ${D3D12_AGILITY_INCLUDE}
)
target_compile_features(DirectXTex PUBLIC cxx_std_17)
target_link_libraries(DirectXTex PUBLIC dxguid ole32)

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
        rdn/CrashHandler.cpp
        src/Util/stb_image.h
        src/Util/stb_image_resize2.h
        rdn/CameraRecorder.cpp
        rdn/CameraRecorder.h
        rdn/CameraPathSimulator.cpp
        rdn/CameraPathSimulator.h

        rdn/Core/DeviceContext.h
        rdn/Core/DeviceContext.cpp
        rdn/Core/ResourceFactory.h
        rdn/Core/GpuProfiler.h
        rdn/Camera/Camera.h
        rdn/Camera/Camera.cpp
        rdn/Editor/Editor.h
        rdn/Editor/Editor.cpp
        rdn/PostProcess/DLSSManager.h
        rdn/PostProcess/DLSSManager.cpp
        rdn/PostProcess/DLSSNRManager.h
        rdn/PostProcess/DLSSNRManager.cpp
        rdn/Raytracing/PassSystem.h
        rdn/Raytracing/PassSystem.cpp
        rdn/Scene/Scene.h
        rdn/Scene/Scene.cpp
        rdn/Scene/AssetLoader.h
        rdn/Scene/AssetLoader.cpp
        rdn/Scene/OmmBuilder.h
        rdn/Scene/OmmBuilder.cpp
        rdn/Renderer_Pipeline.cpp
        rdn/ReuseTextureGen.cpp
        rdn/ReuseTextureGen.h
        rdn/Common.h

        rdn/ThirdPartyImpl.cpp
        rdn/Lighting/LightTreeRefit.h

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
        rdn/planet/heightmap_source.h
        rdn/planet/heightmap_cubemap.h
        rdn/planet/heightmap_cubemap.cpp
        rdn/planet/worker_pool.h
        rdn/planet/worker_pool.cpp
        rdn/planet/tessellator.h
        rdn/planet/tessellator.cpp
        rdn/planet/queue_sync.h
        rdn/planet/queue_sync.cpp
        rdn/planet/generation.h
        rdn/planet/generation.cpp
        rdn/planet/rock_scatter.h
        rdn/planet/rock_scatter.cpp
        rdn/planet/stream_orchestrator.h
        rdn/planet/stream_orchestrator.cpp
        rdn/planet/tlas_builder.h
        rdn/planet/tlas_builder.cpp
        rdn/ocean/OceanCommon.h
        rdn/ocean/OceanQuadtree.h
        rdn/ocean/OceanSystem.h
        rdn/ocean/OceanSystem.cpp

        rdn/minecraft/mc_types.h
        rdn/minecraft/mc_config.h
        rdn/minecraft/mc_inflate.h
        rdn/minecraft/mc_inflate.cpp
        rdn/minecraft/nbt.h
        rdn/minecraft/nbt.cpp
        rdn/minecraft/anvil.h
        rdn/minecraft/anvil.cpp
        rdn/minecraft/block_registry.h
        rdn/minecraft/block_registry.cpp
        rdn/minecraft/mc_zip.h
        rdn/minecraft/mc_zip.cpp
        rdn/minecraft/mc_models.h
        rdn/minecraft/mc_models.cpp
        rdn/minecraft/mc_materials.h
        rdn/minecraft/mc_materials.cpp
        rdn/minecraft/voxel_section.h
        rdn/minecraft/voxel_store.h
        rdn/minecraft/voxel_store.cpp
        rdn/minecraft/voxel_mesher.h
        rdn/minecraft/voxel_mesher.cpp
        rdn/minecraft/mc_bake.h
        rdn/minecraft/mc_placement.h
        rdn/minecraft/mc_omm.h
        rdn/minecraft/mc_omm.cpp
        rdn/minecraft/mc_omm_bake.h
        rdn/minecraft/mc_omm_bake.cpp
        rdn/minecraft/mc_bake.cpp
        rdn/minecraft/lod_tree.h
        rdn/minecraft/lod_tree.cpp
        rdn/minecraft/mc_world.h
        rdn/minecraft/mc_world.cpp
        rdn/minecraft/voxel_streamer.h
        rdn/minecraft/voxel_streamer.cpp
        rdn/Renderer_Minecraft.cpp

        engine/Core/Transform.h
        engine/Input/InputManager.h
        engine/Input/InputManager.cpp
        engine/Scene/GameObject.h
        engine/Scene/SceneDefinition.h
        engine/Scene/SceneManager.h
        engine/Scene/SceneManager.cpp
        engine/Scene/EmissiveCubes.h
        engine/Scene/EmissiveCubes.cpp
        engine/Scene/Ocean.h
        engine/Scene/Ocean.cpp
        engine/Camera/FlyCamController.h
        engine/Camera/FlyCamController.cpp
        engine/EngineApp.h
        engine/EngineApp.cpp

        lib/imgui/imgui.cpp
        lib/imgui/imgui_draw.cpp
        lib/imgui/imgui_tables.cpp
        lib/imgui/imgui_widgets.cpp
        lib/imgui/imgui_impl_dx12.cpp
        lib/imgui/imgui_impl_win32.cpp
)

target_include_directories(Pathtracer PRIVATE
        ${D3D12_AGILITY_INCLUDE}
        ${SL_INCLUDE_DIR}
        ${OMM_SDK_ROOT}/include
        ${CMAKE_SOURCE_DIR}/lib/imgui)

target_compile_definitions(Pathtracer PRIVATE NOMINMAX)

add_library(sl_interposer STATIC IMPORTED GLOBAL)
set_target_properties(sl_interposer PROPERTIES
        IMPORTED_LOCATION           "${SL_LIB_PATH}"
        IMPORTED_LOCATION_DEBUG     "${SL_LIB_PATH}"
        IMPORTED_LOCATION_RELEASE   "${SL_LIB_PATH}"
)
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

target_link_libraries(Pathtracer PRIVATE
        DirectXTex
        sl_interposer
        omm_sdk omm_lz4 omm_xxhash
        d3d12.lib
        dxgi.lib dxguid.lib
        dxcompiler.lib d3dcompiler.lib windowsapp)

# The OMM static library also exports stb_image symbols.
target_link_options(Pathtracer PRIVATE /FORCE:MULTIPLE)

option(PATHTRACER_ENABLE_DLSSNR "Build the native experimental DLSS 5 bridge" ON)
set(DLSSNR_RUNTIME_FILE "" CACHE FILEPATH "User-supplied signed DLSS-NR 310.8.0 DLL (not downloaded or redistributed)")
target_compile_definitions(Pathtracer PRIVATE PATHTRACER_DLSSNR_NATIVE=$<BOOL:${PATHTRACER_ENABLE_DLSSNR}>)
if(PATHTRACER_ENABLE_DLSSNR)
    add_library(DLSSNRBridge SHARED rdn/PostProcess/DLSSNRBridge.cpp rdn/PostProcess/DLSSNRBridge.h)
    target_compile_features(DLSSNRBridge PRIVATE cxx_std_17)
    target_compile_definitions(DLSSNRBridge PRIVATE DLSSNR_BRIDGE_BUILD)
    target_include_directories(DLSSNRBridge PRIVATE "${CMAKE_SOURCE_DIR}/third_party/dlss-ngx/include")
    target_link_libraries(DLSSNRBridge PRIVATE bcrypt.lib)
    set_target_properties(DLSSNRBridge PROPERTIES OUTPUT_NAME "nvngx.dll_royaltracer_nr")
    target_link_libraries(Pathtracer PRIVATE DLSSNRBridge)
    if(DLSSNR_RUNTIME_FILE)
        if(NOT EXISTS "${DLSSNR_RUNTIME_FILE}")
            message(FATAL_ERROR "DLSSNR_RUNTIME_FILE does not exist: ${DLSSNR_RUNTIME_FILE}")
        endif()
        file(SHA256 "${DLSSNR_RUNTIME_FILE}" _nr_hash)
        if(NOT _nr_hash STREQUAL "e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e")
            message(FATAL_ERROR "Unrecognized DLSS-NR binary. This experimental bridge is pinned to the tested signed 310.8.0 runtime.")
        endif()
        add_custom_target(copy_dlssnr_runtime ALL
            COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:Pathtracer>/dlssnr"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${DLSSNR_RUNTIME_FILE}"
                "$<TARGET_FILE_DIR:Pathtracer>/dlssnr/nvngx_dlssnr.dll"
            COMMENT "Staging the user-supplied DLSS-NR runtime")
        add_dependencies(Pathtracer copy_dlssnr_runtime)
    endif()
    message(STATUS "DLSS 5 backend: native experimental bridge (signed 310.8.0)")
endif()
target_link_libraries(Pathtracer PRIVATE wintrust.lib version.lib)

message(STATUS "Streamline include: ${SL_INCLUDE_DIR}")
message(STATUS "Streamline lib    : ${SL_LIB_PATH}")
message(STATUS "Streamline bin    : ${SL_BIN_DIR}")
message(STATUS "Includes dir      : ${INCLUDES_DIR}")

file(GLOB SL_RUNTIME_FILES CONFIGURE_DEPENDS "${SL_BIN_DIR}/*.dll" "${SL_BIN_DIR}/*.license.txt")
if(NOT EXISTS "${SL_BIN_DIR}/sl.interposer.dll" OR NOT EXISTS "${SL_BIN_DIR}/nvngx_dlssd.dll")
    message(FATAL_ERROR "Incomplete Streamline runtime at ${SL_BIN_DIR}")
endif()
# Stage SDK runtimes after assets to replace legacy DLLs.
add_custom_target(copy_assets ALL
        COMMENT "Copying include/ and shaders/ to runtime directory"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${INCLUDES_DIR}"
        "$<TARGET_FILE_DIR:Pathtracer>"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SHADERS_DIR}"
        "$<TARGET_FILE_DIR:Pathtracer>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${SL_RUNTIME_FILES} "$<TARGET_FILE_DIR:Pathtracer>"
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
        rdn/planet/heightmap_procedural.h
        rdn/planet/heightmap_procedural.cpp
        rdn/planet/worker_pool.h
        rdn/planet/worker_pool.cpp
        rdn/planet/tessellator.h
        rdn/planet/tessellator.cpp
        rdn/planet/planet_tests.cpp
)
target_compile_features(PlanetTests PRIVATE cxx_std_23)

add_executable(MinecraftTests
        rdn/minecraft/mc_types.h
        rdn/minecraft/mc_inflate.h
        rdn/minecraft/mc_inflate.cpp
        rdn/minecraft/nbt.h
        rdn/minecraft/nbt.cpp
        rdn/minecraft/anvil.h
        rdn/minecraft/anvil.cpp
        rdn/minecraft/block_registry.h
        rdn/minecraft/block_registry.cpp
        rdn/minecraft/mc_zip.h
        rdn/minecraft/mc_zip.cpp
        rdn/minecraft/mc_models.h
        rdn/minecraft/mc_models.cpp
        rdn/minecraft/voxel_section.h
        rdn/minecraft/voxel_store.h
        rdn/minecraft/voxel_store.cpp
        rdn/minecraft/voxel_mesher.h
        rdn/minecraft/voxel_mesher.cpp
        rdn/minecraft/mc_bake.h
        rdn/minecraft/mc_placement.h
        rdn/minecraft/mc_omm.h
        rdn/minecraft/mc_omm.cpp
        rdn/minecraft/mc_bake.cpp
        rdn/minecraft/mc_materials.h
        rdn/minecraft/mc_materials.cpp
        rdn/minecraft/lod_tree.h
        rdn/minecraft/lod_tree.cpp
        rdn/minecraft/mc_world.h
        rdn/minecraft/mc_world.cpp
        rdn/planet/worker_pool.h
        rdn/planet/worker_pool.cpp
        tests/MinecraftTests.cpp
)
target_include_directories(MinecraftTests PRIVATE ${CMAKE_SOURCE_DIR}/rdn)
target_link_libraries(MinecraftTests PRIVATE DirectXTex)
target_compile_features(MinecraftTests PRIVATE cxx_std_23)
target_compile_definitions(MinecraftTests PRIVATE NOMINMAX)

# Opt-in: LOD geometry counts and DXR traversal timing.
get_target_property(MINECRAFT_SURVEY_SOURCES MinecraftTests SOURCES)
list(FILTER MINECRAFT_SURVEY_SOURCES EXCLUDE REGEX "tests/MinecraftTests.cpp$")
add_executable(MinecraftBvhSurvey EXCLUDE_FROM_ALL
    tests/MinecraftBvhSurvey.cpp rdn/planet/tlas_builder.cpp rdn/planet/blas_pool.cpp ${MINECRAFT_SURVEY_SOURCES})
target_include_directories(MinecraftBvhSurvey PRIVATE ${CMAKE_SOURCE_DIR}/rdn ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(MinecraftBvhSurvey PRIVATE DirectXTex d3d12 dxgi)
target_compile_features(MinecraftBvhSurvey PRIVATE cxx_std_23)
target_compile_definitions(MinecraftBvhSurvey PRIVATE NOMINMAX)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME PlanetTests COMMAND PlanetTests)
    add_test(NAME MinecraftTests COMMAND MinecraftTests)
endif()
