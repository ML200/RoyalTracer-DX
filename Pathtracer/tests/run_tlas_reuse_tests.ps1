param([string]$OutputDirectory = (Join-Path $PSScriptRoot '../out/tlas-reuse-tests'))
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path $OutputDirectory).Path
& "$projectRoot/include/dxc.exe" -T cs_6_5 -E main -HV 2021 -O3 `
    "$PSScriptRoot/TlasReuseTests.hlsl" -Fo "$OutputDirectory/tlas-reuse.dxil"
if ($LASTEXITCODE -ne 0) { throw 'TLAS shader compilation failed' }
& cl.exe /nologo /std:c++17 /EHsc /O2 /MD /Gy /DNOMINMAX `
    "/I$projectRoot/include" "/I$projectRoot/rdn" `
    "$PSScriptRoot/TlasReuseTests.cpp" "$projectRoot/rdn/planet/tlas_builder.cpp" `
    "$projectRoot/rdn/planet/blas_pool.cpp" "/Fo$OutputDirectory/" "/Fe$OutputDirectory/TlasReuseTests.exe" `
    /link /OPT:REF d3d12.lib dxgi.lib
if ($LASTEXITCODE -ne 0) { throw 'TLAS test compilation failed. Use a VS Developer shell.' }
& "$OutputDirectory/TlasReuseTests.exe" "$OutputDirectory/tlas-reuse.dxil"
if ($LASTEXITCODE -ne 0) { throw 'TLAS reuse regression failed' }
