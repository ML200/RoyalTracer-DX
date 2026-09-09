param([string]$OutputDirectory = (Join-Path $PSScriptRoot '../out/light-tree-tests'))
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path $OutputDirectory).Path
& "$projectRoot/include/dxc.exe" -T cs_6_6 -E main -HV 2021 -enable-16bit-types -O3 `
    -I "$projectRoot/shaders" -I "$projectRoot/include" `
    "$PSScriptRoot/LightTreeGpuTests.hlsl" -Fo "$OutputDirectory/light-tree.dxil"
if ($LASTEXITCODE -ne 0) { throw 'Light-tree shader compilation failed' }
& cl.exe /nologo /std:c++17 /EHsc /O2 /DLT_ENABLE_LOGS=0 /DLT_ENABLE_TIMING=0 `
    "/I$projectRoot/include" "/I$projectRoot/rdn" "/I$projectRoot/DirectXTex" `
    "$PSScriptRoot/LightTreeGpuTests.cpp" "/Fo$OutputDirectory/LightTreeGpuTests.obj" `
    "/Fe$OutputDirectory/LightTreeGpuTests.exe" /link d3d12.lib dxgi.lib
if ($LASTEXITCODE -ne 0) { throw 'Light-tree runner compilation failed. Use a VS Developer shell.' }
& "$OutputDirectory/LightTreeGpuTests.exe" $OutputDirectory
if ($LASTEXITCODE -ne 0) { throw 'Light-tree regression failed' }
