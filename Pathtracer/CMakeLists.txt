cmake_minimum_required(VERSION 3.25.2)
set(CMAKE_SYSTEM_VERSION "10.0.22621.0")
project(Pathtracer)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
        rdn/ResourceStateTracker.h

        rdn/Core/DeviceContext.h
        rdn/Core/DeviceContext.cpp
        rdn/Core/ResourceFactory.h
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
