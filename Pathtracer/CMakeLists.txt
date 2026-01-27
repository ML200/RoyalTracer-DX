cmake_minimum_required(VERSION 3.25.2)
project(Pathtracer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ───────────────────────── DirectX Agility SDK ─────────────────────────────
set(D3D12_AGILITY_ROOT "${CMAKE_SOURCE_DIR}/Microsoft.Direct3D.D3D12.1.717.1-preview/build/native")
set(D3D12_AGILITY_INCLUDE "${D3D12_AGILITY_ROOT}/include")
set(D3D12_AGILITY_LIB "${D3D12_AGILITY_ROOT}/bin/x64")

# ───────────────────────────── Streamline paths ──────────────────────────────
set(SL_ROOT        "${CMAKE_SOURCE_DIR}/streamline")
set(SL_INCLUDE_DIR "${SL_ROOT}/include")
set(SL_LIB_PATH    "${SL_ROOT}/lib/x64/sl.interposer.lib")
set(SL_BIN_DIR     "${SL_ROOT}/bin/x64")

# ────────────────────────── extra assets to copy ─────────────────────────────
set(INCLUDES_DIR   "${CMAKE_SOURCE_DIR}/include")
set(SHADERS_DIR   "${CMAKE_SOURCE_DIR}/shaders")


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
)

# ───────────────────────── include directories ───────────────────────────────
target_include_directories(Pathtracer PRIVATE
        ${D3D12_AGILITY_INCLUDE}
        ${SL_INCLUDE_DIR})

# ──────────────── Streamline import‐lib as an IMPORTED static library ────────
add_library(sl_interposer STATIC IMPORTED GLOBAL)
set_target_properties(sl_interposer PROPERTIES
        IMPORTED_LOCATION           "${SL_LIB_PATH}"
        IMPORTED_LOCATION_DEBUG     "${SL_LIB_PATH}"
        IMPORTED_LOCATION_RELEASE   "${SL_LIB_PATH}"
)

# ───────────────────────── link libraries ────────────────────────────────────
target_link_libraries(Pathtracer PRIVATE
        DirectXTex
        sl_interposer
        d3d12.lib
        dxgi.lib dxguid.lib
        dxcompiler.lib d3dcompiler.lib)


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
        "$<TARGET_FILE_DIR:Pathtracer>"
        COMMENT "Copying Agility D3D12Core.dll to output folder."
)