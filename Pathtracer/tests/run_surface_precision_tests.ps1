param([string]$OutputDirectory = (Join-Path $PSScriptRoot '../out/surface-precision-tests'))
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path $OutputDirectory).Path
& "$projectRoot/include/dxc.exe" -T cs_6_6 -E main -HV 2021 -enable-16bit-types -O3 `
    -I "$projectRoot/shaders" "$PSScriptRoot/SurfacePrecisionTests.hlsl" -Fo "$OutputDirectory/surface-precision.dxil"
if ($LASTEXITCODE -ne 0) { throw 'Surface-precision shader compilation failed' }
& cl.exe /nologo /std:c++17 /EHsc /O2 /MD /Gy /DNOMINMAX `
    "/I$projectRoot/include" "/I$projectRoot/rdn" `
    "$PSScriptRoot/SurfacePrecisionTests.cpp" "$projectRoot/rdn/planet/tlas_builder.cpp" `
    "$projectRoot/rdn/planet/blas_pool.cpp" "/Fo$OutputDirectory/" "/Fe$OutputDirectory/SurfacePrecisionTests.exe" `
    /link /OPT:REF d3d12.lib dxgi.lib
if ($LASTEXITCODE -ne 0) { throw 'Surface-precision runner compilation failed. Use a VS Developer shell.' }
& "$OutputDirectory/SurfacePrecisionTests.exe" "$OutputDirectory/surface-precision.dxil"
if ($LASTEXITCODE -ne 0) { throw 'Surface-precision regression failed' }
