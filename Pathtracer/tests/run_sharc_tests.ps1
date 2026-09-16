param(
    [string]$OutputDirectory = (Join-Path $env:TEMP 'RoyalTracer-SharcTests'),
    [string]$ShaderDirectory = (Join-Path $PSScriptRoot '../shaders')
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$dxc = Join-Path $projectRoot 'include/dxc.exe'
foreach ($entry in @('prepare', 'fill', 'resolve', 'query', 'eraseTop', 'benchmark', 'guideFill', 'guideQuery', 'materialCheck', 'materialBenchmark', 'liteCheck', 'legacyDupCheck', 'materialSamplingCheck')) {
    $readOnly = if ($entry -in @('query', 'benchmark')) { @('-D', 'SHARC_READ_ONLY=1') } else { @() }
    & $dxc @readOnly -I $ShaderDirectory -T cs_6_6 -E $entry -HV 2021 -enable-16bit-types -O3 -WX `
        (Join-Path $PSScriptRoot 'SharcGpuTests.hlsl') -Fo (Join-Path $OutputDirectory "$entry.dxil")
    if ($LASTEXITCODE -ne 0) { throw "Shader compilation failed: $entry" }
}
& cl.exe /nologo /std:c++17 /EHsc /O2 "/I$ShaderDirectory" (Join-Path $PSScriptRoot 'SharcGpuTests.cpp') `
    "/Fo$OutputDirectory/SharcGpuTests.obj" "/Fe$OutputDirectory/SharcGpuTests.exe" `
    /link d3d12.lib dxgi.lib
if ($LASTEXITCODE -ne 0) { throw 'GPU runner compilation failed. Use a VS Developer shell.' }
& (Join-Path $OutputDirectory 'SharcGpuTests.exe') $OutputDirectory
if ($LASTEXITCODE -ne 0) { throw 'SHaRC GPU regression failed' }
