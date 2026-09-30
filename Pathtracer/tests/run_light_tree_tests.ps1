param([string]$OutputDirectory = (Join-Path $PSScriptRoot '../out/light-tree-tests'),
    [string]$ShaderDirectory = (Join-Path $PSScriptRoot '../shaders'), [switch]$Benchmark, [switch]$HotCellBenchmark, [switch]$LodTest,
    [string]$TestShader = (Join-Path $PSScriptRoot 'LightTreeGpuTests.hlsl'), [switch]$GridReviewTest, [switch]$SurfaceTest, [switch]$ColdLodTest)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path $OutputDirectory).Path
& "$projectRoot/include/dxc.exe" -T cs_6_6 -E main -HV 2021 -enable-16bit-types -O3 `
    -I "$ShaderDirectory" -I "$projectRoot/include" `
    "$TestShader" -Fo "$OutputDirectory/light-tree.dxil"
if ($LASTEXITCODE -ne 0) { throw 'Light-tree shader compilation failed' }
& "$projectRoot/include/dxc.exe" -T cs_6_6 -E main -HV 2021 -enable-16bit-types -O3 `
    -I "$ShaderDirectory" -I "$projectRoot/include" `
    "$ShaderDirectory/Pass_light_learning_v8.hlsl" -Fo "$OutputDirectory/light-learning.dxil"
if ($LASTEXITCODE -ne 0) { throw 'Light-learning shader compilation failed' }

& cl.exe /nologo /std:c++17 /EHsc /O2 /DLT_ENABLE_LOGS=0 /DLT_ENABLE_TIMING=0 `
    "/I$projectRoot/include" "/I$projectRoot/rdn" "/I$projectRoot/DirectXTex" `
    "$PSScriptRoot/LightTreeGpuTests.cpp" "/Fo$OutputDirectory/LightTreeGpuTests.obj" `
    "/Fe$OutputDirectory/LightTreeGpuTests.exe" /link d3d12.lib dxgi.lib
if ($LASTEXITCODE -ne 0) { throw 'Light-tree runner compilation failed. Use a VS Developer shell.' }
if($HotCellBenchmark) { & "$OutputDirectory/LightTreeGpuTests.exe" $OutputDirectory --hot-benchmark }
elseif($Benchmark) { & "$OutputDirectory/LightTreeGpuTests.exe" $OutputDirectory --benchmark }
elseif($LodTest) { & "$OutputDirectory/LightTreeGpuTests.exe" $OutputDirectory --lod }
elseif($GridReviewTest) { & "$OutputDirectory/LightTreeGpuTests.exe" $OutputDirectory --grid-review }
elseif($SurfaceTest) { & "$OutputDirectory/LightTreeGpuTests.exe" $OutputDirectory --surface }
elseif($ColdLodTest) { & "$OutputDirectory/LightTreeGpuTests.exe" $OutputDirectory --cold-lod }
else { & "$OutputDirectory/LightTreeGpuTests.exe" $OutputDirectory }
if ($LASTEXITCODE -ne 0) { throw 'Light-tree regression failed' }
