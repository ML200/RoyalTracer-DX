cmake_minimum_required(VERSION 3.25.2)
project(Pathtracer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Locate the Windows SDK for DirectX
set(WINDOWS_SDK_VERSION "10.0")
# Set the paths for DirectX headers and libraries
set(DIRECTX_SDK_INCLUDE "C:/Program Files (x86)/Windows Kits/10/Include/${WINDOWS_SDK_VERSION}/um")
set(DIRECTX_SDK_LIB "C:/Program Files (x86)/Windows Kits/10/Lib/${WINDOWS_SDK_VERSION}/um/x64")



add_executable(Pathtracer WIN32 src/main.cpp
        src/Objects/SceneObject.cpp
        src/Objects/SceneObject.h
        src/Scene/Scene.cpp
        src/Scene/Scene.h
        src/Components/Material.h
        src/Components/Transform.cpp
        src/Components/Transform.h
        src/Math/Vector3.h
        src/Math/Quaternion.h
        src/Components/Mesh.cpp
        src/Components/Mesh.h
        src/Components/Vertex.cpp
        src/Components/Vertex.h
        src/Math/Vector2.h
        src/Math/Vector4.h
        src/Util/ObjLoader.h src/Math/Helpers.h
        src/Util/D3DVertex.h
        src/Renderer/inc/DX12LibPCH.h
        src/Renderer/inc/CommandQueue.h
        src/Renderer/inc/Application.h
        src/Renderer/inc/Events.h
        src/Renderer/inc/Game.h
        src/Renderer/inc/Helpers.h
        src/Renderer/inc/HighResolutionClock.h
        src/Renderer/inc/KeyCodes.h
        src/Renderer/inc/Window.h
        src/Renderer/src/Application.cpp
        src/Renderer/src/CommandQueue.cpp
        src/Renderer/src/DX12LibPCH.cpp
        src/Renderer/src/Game.cpp
        src/Renderer/src/HighResolutionClock.cpp
        src/Renderer/src/Window.cpp
        src/Renderer/resource.h
        src/Tutorial2.cpp
        src/Tutorial2.h)

# Include directories
include_directories(${DIRECTX_SDK_INCLUDE})
target_include_directories(Pathtracer PRIVATE ${DIRECTX_SDK_INCLUDE} ${FASTGLTF_INCLUDE_DIR})


target_link_directories(Pathtracer PRIVATE ${DIRECTX_SDK_LIB} ${FASTGLTF_LIB_DIR})
target_link_libraries(Pathtracer PRIVATE d3d12.lib dxgi.lib dxguid.lib d3dcompiler.lib shlwapi.lib)




