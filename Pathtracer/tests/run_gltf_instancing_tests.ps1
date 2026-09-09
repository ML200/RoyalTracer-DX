param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '../out/gltf-instancing-tests'),
    [string]$DirectXTexLibrary = (Join-Path $PSScriptRoot '../cmake-build-relwithdebinfo-visual-studio/DirectXTex.lib')
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path $OutputDirectory).Path
& "$projectRoot/include/dxc.exe" -T cs_6_6 -E main -HV 2021 -enable-16bit-types -O3 `
    -I "$projectRoot/shaders" -I "$projectRoot/include" `
    "$PSScriptRoot/GltfInstancingGpuTests.hlsl" -Fo "$OutputDirectory/gltf-normals.dxil"
if ($LASTEXITCODE -ne 0) { throw 'Instancing GPU shader compilation failed' }
if (!(Test-Path -LiteralPath $DirectXTexLibrary)) {
    throw 'Build the Release/RelWithDebInfo DirectXTex target first, or pass -DirectXTexLibrary.'
}
$agility = "$projectRoot/microsoft.direct3d.d3d12.1.719.0-preview/build/native/include"
$sources = @(
    "$PSScriptRoot/GltfInstancingTests.cpp",
    "$projectRoot/rdn/Scene/AssetLoader.cpp",
    "$projectRoot/rdn/Scene/Scene.cpp",
    "$projectRoot/rdn/Scene/OmmBuilder.cpp",
    "$projectRoot/rdn/ThirdPartyImpl.cpp"
)
& cl.exe /nologo /std:c++latest /EHsc /O2 /MD /Gy /DNOMINMAX /DLT_ENABLE_LOGS=0 /DLT_ENABLE_TIMING=0 `
    "/I$agility" "/I$projectRoot/include" "/I$projectRoot/rdn" "/I$projectRoot/DirectXTex" `
    "/I$projectRoot/third_party/omm-sdk/include" @sources "/Fo$OutputDirectory/" `
    "/Fe$OutputDirectory/GltfInstancingTests.exe" /link /OPT:REF /FORCE:MULTIPLE `
    "$DirectXTexLibrary" "$projectRoot/third_party/omm-sdk/lib/omm-lib.lib" `
    "$projectRoot/third_party/omm-sdk/lib/lz4.lib" "$projectRoot/third_party/omm-sdk/lib/xxhash.lib" `
    d3d12.lib dxgi.lib dxguid.lib ole32.lib windowscodecs.lib
if ($LASTEXITCODE -ne 0) { throw 'Instancing test compilation failed. Use a VS Developer shell.' }
& "$OutputDirectory/GltfInstancingTests.exe" "$OutputDirectory/fixtures"
if ($LASTEXITCODE -ne 0) { throw 'Instancing regression failed' }

# Compile production entry points that consume surface normals and light samples.
foreach ($shader in @('Pass_pt_v8.hlsl', 'Pass_shading_v8.hlsl')) {
    $shaderTarget = if ($shader -eq 'Pass_shading_v8.hlsl') { @('-T', 'cs_6_9', '-E', 'main') } else { @('-T', 'lib_6_9') }
    & "$projectRoot/include/dxc.exe" @shaderTarget -HV 2021 -enable-16bit-types -O3 -D MAX_REGS=96 `
        -I "$projectRoot/shaders" -I "$projectRoot/include" `
        "$projectRoot/shaders/$shader" -Fo "$OutputDirectory/$shader.dxil"
    if ($LASTEXITCODE -ne 0) { throw "Production shader compilation failed: $shader" }
}
