param(
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '../out/gpu-profiler-tests')
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path $OutputDirectory).Path
$agility = "$projectRoot/microsoft.direct3d.d3d12.1.719.0-preview/build/native/include"
& cl.exe /nologo /std:c++latest /EHsc /O2 /MD /Gy /DNOMINMAX `
    "/I$agility" "/I$projectRoot/include" "/I$projectRoot/rdn" "/I$projectRoot/DirectXTex" `
    "$PSScriptRoot/GpuProfilerTests.cpp" "/Fo$OutputDirectory/" `
    "/Fe$OutputDirectory/GpuProfilerTests.exe" /link /OPT:REF d3d12.lib dxgi.lib
if ($LASTEXITCODE -ne 0) { throw 'GPU profiler test compilation failed. Use a VS Developer shell.' }
& "$OutputDirectory/GpuProfilerTests.exe"
if ($LASTEXITCODE -ne 0) { throw 'GPU profiler regression failed' }
