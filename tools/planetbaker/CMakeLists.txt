cmake_minimum_required(VERSION 3.27)

project(planet_bake LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "" FORCE)
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

#====================================
# CPM bootstrap
#====================================
set(CPM_DOWNLOAD_VERSION 0.40.5)
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT EXISTS ${CPM_DOWNLOAD_LOCATION})
    message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
    file(DOWNLOAD
        https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
        ${CPM_DOWNLOAD_LOCATION}
        TLS_VERIFY ON)
endif()
include(${CPM_DOWNLOAD_LOCATION})

#====================================
# Third-party dependencies
#====================================
CPMAddPackage(
    NAME glfw
    GITHUB_REPOSITORY glfw/glfw
    GIT_TAG 3.4
    OPTIONS
        "GLFW_BUILD_DOCS OFF"
        "GLFW_BUILD_TESTS OFF"
        "GLFW_BUILD_EXAMPLES OFF"
        "GLFW_INSTALL OFF"
)

CPMAddPackage(
    NAME glew
    GITHUB_REPOSITORY Perlmint/glew-cmake
    GIT_TAG glew-cmake-2.2.0
    OPTIONS
        "glew-cmake_BUILD_SHARED OFF"
        "ONLY_LIBS ON"
)

CPMAddPackage(
    NAME imgui
    GITHUB_REPOSITORY ocornut/imgui
    GIT_TAG docking
    DOWNLOAD_ONLY YES
)

if(imgui_ADDED)
    add_library(imgui STATIC
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/imgui_demo.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )
    target_include_directories(imgui PUBLIC
        ${imgui_SOURCE_DIR}
        ${imgui_SOURCE_DIR}/backends
    )
    target_link_libraries(imgui PUBLIC glfw OpenGL::GL)
    set_target_properties(imgui PROPERTIES FOLDER "third_party")
endif()

CPMAddPackage(
    NAME glm
    GITHUB_REPOSITORY g-truc/glm
    GIT_TAG 1.0.1
)

CPMAddPackage(
    NAME json
    GITHUB_REPOSITORY nlohmann/json
    GIT_TAG v3.11.3
    OPTIONS "JSON_BuildTests OFF"
)

CPMAddPackage(
    NAME doctest
    GITHUB_REPOSITORY doctest/doctest
    GIT_TAG v2.4.11
)

#====================================
# Main executable
#====================================
add_executable(planet_bake
    src/main.cpp
    src/gpu/device_info.cu
    src/gpu/preview_field.cu
    src/render/sphere_mesh.cpp
    src/render/shader.cpp
    src/render/camera.cpp
    src/render/preview.cpp
)

target_include_directories(planet_bake PRIVATE src)

target_link_libraries(planet_bake PRIVATE
    imgui
    glfw
    libglew_static
    glm::glm
    nlohmann_json::nlohmann_json
    OpenGL::GL
    CUDA::cudart
)

target_compile_definitions(planet_bake PRIVATE GLEW_STATIC)

set_target_properties(planet_bake PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

if(MSVC)
    target_compile_options(planet_bake PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/W4 /permissive- /Zc:__cplusplus /utf-8>
    )
    target_compile_definitions(planet_bake PRIVATE
        _CRT_SECURE_NO_WARNINGS
        NOMINMAX
        WIN32_LEAN_AND_MEAN
    )
else()
    target_compile_options(planet_bake PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
    )
endif()

#====================================
# Tests
#====================================
enable_testing()
add_executable(planet_bake_tests
    tests/sanity_test.cpp
)
target_link_libraries(planet_bake_tests PRIVATE doctest::doctest)
target_include_directories(planet_bake_tests PRIVATE src)
add_test(NAME planet_bake_tests COMMAND planet_bake_tests)
