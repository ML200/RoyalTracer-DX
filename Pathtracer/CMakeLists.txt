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
        src/Util/ObjLoader.h)

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

# ───────────────────── copy Streamline runtime DLLs ──────────────────────────
#add_custom_command(TARGET Pathtracer POST_BUILD
#        COMMAND ${CMAKE_COMMAND} -E copy_if_different
#        "${SL_BIN_DIR}/sl.interposer.dll"
#        "${SL_BIN_DIR}/sl.common.dll"
#        "${SL_BIN_DIR}/sl.dlss_g.dll"
#        "${SL_BIN_DIR}/sl.dlss_rr.dll"
#        $<TARGET_FILE_DIR:Pathtracer>)

# ─────────────────────────── status messages ────────────────────────────────
message(STATUS "Streamline include: ${SL_INCLUDE_DIR}")
message(STATUS "Streamline lib    : ${SL_LIB_PATH}")
message(STATUS "Streamline bin    : ${SL_BIN_DIR}")
