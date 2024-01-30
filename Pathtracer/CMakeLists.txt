cmake_minimum_required(VERSION 3.26)
project(Pathtracer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Locate the Windows SDK for DirectX
set(WINDOWS_SDK_VERSION "10.0") # Replace with your Windows SDK version if different
add_executable(Pathtracer main.cpp
        src/Objects/sceneObject.cpp
        src/Objects/sceneObject.h
        src/Renderer/renderer.cpp
        src/Renderer/renderer.h
        src/Scene/scene.cpp
        src/Scene/scene.h
        src/Components/Triangle.cpp
        src/Components/Triangle.h
        src/Components/Material.cpp
        src/Components/Material.h
        src/Components/Transform.cpp
        src/Components/Transform.h
        src/Math/Vector3.cpp
        src/Math/Vector3.h
        src/Math/Quaterion.cpp
        src/Math/Quaterion.h)


# Set the paths for DirectX headers and libraries
set(DIRECTX_SDK_INCLUDE "C:/Program Files (x86)/Windows Kits/10/Include/${WINDOWS_SDK_VERSION}/um") # Replace with your DirectX SDK include path
set(DIRECTX_SDK_LIB "C:/Program Files (x86)/Windows Kits/10/Lib/${WINDOWS_SDK_VERSION}/um/x64") # Replace with your DirectX SDK lib path (use x86 for 32-bit)

# Include directories
include_directories(${DIRECTX_SDK_INCLUDE})

# Link libraries
target_link_directories(Pathtracer PRIVATE ${DIRECTX_SDK_LIB})
target_link_libraries(Pathtracer d3d12.lib dxgi.lib)


