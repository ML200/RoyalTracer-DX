param([string]$OutputDirectory = (Join-Path $PSScriptRoot '../out/ocean-tests'))
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path $OutputDirectory).Path
& cl.exe /nologo /std:c++17 /EHsc /O2 /MD /DNOMINMAX "/I$root/DirectXTex" "/I$root/include" "/I$root/rdn" `
    "$PSScriptRoot/OceanTests.cpp" "/Fo$OutputDirectory/OceanTests.obj" "/Fe$OutputDirectory/OceanTests.exe"
if ($LASTEXITCODE -ne 0) { throw 'Ocean CPU test build failed; use a VS Developer shell.' }
& "$OutputDirectory/OceanTests.exe"
if ($LASTEXITCODE -ne 0) { throw 'Ocean CPU regression failed' }

$entries = @('OceanFftH','OceanFftV','OceanMip')
foreach ($entry in $entries) {
    & "$root/include/dxc.exe" -T cs_6_6 -E $entry -HV 2021 -enable-16bit-types -O3 -WX `
        -D OCEAN_FFT_SIZE=16 -D OCEAN_FFT_LOG2=4 "$root/shaders/Ocean_Sim_v8.hlsl" -Fo "$OutputDirectory/small-$entry.dxil"
    if ($LASTEXITCODE -ne 0) { throw "Ocean GPU shader build failed: $entry" }
}
& cl.exe /nologo /std:c++17 /EHsc /O2 /MD /DNOMINMAX "/I$root/rdn" "/I$root/include" `
    "$PSScriptRoot/OceanSimulationGpuTests.cpp" "/Fo$OutputDirectory/OceanSimulationGpuTests.obj" `
    "/Fe$OutputDirectory/OceanSimulationGpuTests.exe" /link d3d12.lib dxgi.lib
if ($LASTEXITCODE -ne 0) { throw 'Ocean GPU runner build failed' }
& "$OutputDirectory/OceanSimulationGpuTests.exe" $OutputDirectory
if ($LASTEXITCODE -ne 0) { throw 'Ocean GPU regression failed' }

