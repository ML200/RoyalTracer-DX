cmake_minimum_required(VERSION 3.25.2)
project(Pathtracer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ───────────────────────── Windows / DirectX SDK ─────────────────────────────
set(WINDOWS_SDK_VERSION "10.0.22621.0")
set(WINDOWS_SDK_ROOT    "C:/Program Files (x86)/Windows Kits/10")

set(DIRECTX_SDK_INCLUDE "${WINDOWS_SDK_ROOT}/Include/${WINDOWS_SDK_VERSION}/um")
set(DIRECTX_SDK_LIB     "${WINDOWS_SDK_ROOT}/Lib/${WINDOWS_SDK_VERSION}/um/x64")

# ───────────────────────────── Streamline paths ──────────────────────────────
set(SL_ROOT        "${CMAKE_SOURCE_DIR}/streamline")
set(SL_INCLUDE_DIR "${SL_ROOT}/include")
set(SL_LIB_PATH    "${SL_ROOT}/lib/x64/sl.interposer.lib")   # MUST exist
set(SL_BIN_DIR     "${SL_ROOT}/bin/x64")

# ────────────────────────── extra assets to copy ─────────────────────────────
# Every build: mirror the **contents** of the `includes` folder next to the exe
set(INCLUDES_DIR   "${CMAKE_SOURCE_DIR}/include")
set(SHADERS_DIR   "${CMAKE_SOURCE_DIR}/shaders")

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
        src/Components/Vertex.h
        src/Util/ObjLoader.h
)

# ───────────────────────── include directories ───────────────────────────────
target_include_directories(Pathtracer PRIVATE
        ${DIRECTX_SDK_INCLUDE}
        ${SL_INCLUDE_DIR})

# ──────────────── Streamline import‑lib as an IMPORTED static library ────────
add_library(sl_interposer STATIC IMPORTED GLOBAL)
set_target_properties(sl_interposer PROPERTIES
        IMPORTED_LOCATION           "${SL_LIB_PATH}"
        IMPORTED_LOCATION_DEBUG     "${SL_LIB_PATH}"
        IMPORTED_LOCATION_RELEASE   "${SL_LIB_PATH}"
)

# ───────────────────────── link libraries ────────────────────────────────────
target_link_libraries(Pathtracer PRIVATE
        sl_interposer
        d3d12.lib dxgi.lib dxguid.lib
        dxcompiler.lib d3dcompiler.lib)

# ────────────────── copy the *contents* of 'includes' on every build ─────────
add_custom_command(
        TARGET Pathtracer
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${INCLUDES_DIR}"
        "$<TARGET_FILE_DIR:Pathtracer>"
        COMMENT "Copying contents of 'includes' into build output directory."
)

# ────────────────── copy the *contents* of 'shaders' (new shaders) on every build ─────────
add_custom_command(
        TARGET Pathtracer
        POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SHADERS_DIR}"
        "$<TARGET_FILE_DIR:Pathtracer>"
        COMMENT "Copying contents of 'shaders' into build output directory."
)

# ─────────────────────────── status messages ────────────────────────────────
message(STATUS "Streamline include: ${SL_INCLUDE_DIR}")
message(STATUS "Streamline lib    : ${SL_LIB_PATH}")
message(STATUS "Streamline bin    : ${SL_BIN_DIR}")
message(STATUS "Includes dir      : ${INCLUDES_DIR}")

# ────────────── copy‐assets target ──────────────
# This target will always run when you build the project,
# and will copy include/ and shaders/ into your exe folder.
add_custom_target(copy_assets ALL
        COMMENT "Copying include/ and shaders/ to runtime directory"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${INCLUDES_DIR}"
        "$<TARGET_FILE_DIR:Pathtracer>"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SHADERS_DIR}"
        "$<TARGET_FILE_DIR:Pathtracer>"
)

# Make sure the exe is built before we try to copy into its folder
add_dependencies(copy_assets Pathtracer)

