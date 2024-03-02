cmake_minimum_required(VERSION 3.25.2)
project(Pathtracer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Locate the Windows SDK for DirectX
# Set the Windows SDK version
set(WINDOWS_SDK_VERSION "10.0.18362.0")

# Set the path to the Windows SDK
set(WINDOWS_SDK_ROOT "C:/Program Files (x86)/Windows Kits/10")

# Set the CMAKE_SYSROOT variable to point to the Windows SDK
set(CMAKE_SYSROOT "${WINDOWS_SDK_ROOT}/${WINDOWS_SDK_VERSION}")

# Set the paths for DirectX headers and libraries
set(DIRECTX_SDK_INCLUDE "C:/Program Files (x86)/Windows Kits/10/Include/${WINDOWS_SDK_VERSION}/um")
set(DIRECTX_SDK_LIB "C:/Program Files (x86)/Windows Kits/10/Lib/${WINDOWS_SDK_VERSION}/um/x64")

set(SL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/lib/nv_streamline/include")
set(SL_LIBRARY_DIR "${CMAKE_SOURCE_DIR}/lib/nv_streamline/lib/x64")



add_executable(Pathtracer WIN32
        rdn/Main.cpp
        rdn/nv_helpers_dx12/BottomLevelASGenerator.cpp
        rdn/nv_helpers_dx12/RaytracingPipelineGenerator.cpp
        rdn/nv_helpers_dx12/RootSignatureGenerator.cpp
        rdn/nv_helpers_dx12/ShaderBindingTableGenerator.cpp
        rdn/nv_helpers_dx12/TopLevelASGenerator.cpp
        rdn/nv_helpers_dx12/BottomLevelASGenerator.h
        rdn/nv_helpers_dx12/RaytracingPipelineGenerator.h
        rdn/nv_helpers_dx12/RootSignatureGenerator.h
        rdn/nv_helpers_dx12/ShaderBindingTableGenerator.h
        rdn/nv_helpers_dx12/TopLevelASGenerator.h
        rdn/Renderer.cpp
        rdn/DXSample.cpp
        rdn/manipulator.cpp
        rdn/stdafx.cpp
        rdn/Win32Application.cpp
        rdn/Renderer.h
        rdn/DXSample.h
        rdn/manipulator.h
        rdn/stdafx.h
        rdn/Win32Application.h
        src/Components/Vertex.h
        src/Util/ObjLoader.h


)

# Include directories
include_directories(${DIRECTX_SDK_INCLUDE})
target_include_directories(Pathtracer PRIVATE ${DIRECTX_SDK_INCLUDE} ${FASTGLTF_INCLUDE_DIR} ${SL_INCLUDE_DIR})

target_link_directories(Pathtracer PRIVATE ${DIRECTX_SDK_LIB} ${FASTGLTF_LIB_DIR})
target_link_libraries(Pathtracer PRIVATE d3d12.lib dxgi.lib dxguid.lib dxcompiler.lib d3dcompiler.lib shlwapi.lib ${SL_LIBRARY_DIR}/sl.interposer.lib)




