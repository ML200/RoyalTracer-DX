param([string]$OutputDirectory = (Join-Path $PSScriptRoot '../out/cumulus/gpu'),[int]$Width=640,[int]$Height=360,[switch]$Reference,[switch]$LightingStudy,[switch]$Benchmark,[switch]$Restored,[switch]$DetailStudy,[switch]$TwilightStudy,[switch]$AtmosphereStudy,[switch]$IntegrationStudy,[switch]$TemporalStudy,[switch]$DensityParity,[switch]$DensityCacheStudy,[switch]$Cached)
$ErrorActionPreference = 'Stop'
if (([int]$Reference.IsPresent + [int]$LightingStudy.IsPresent + [int]$Benchmark.IsPresent + [int]$Restored.IsPresent + [int]$DetailStudy.IsPresent + [int]$TwilightStudy.IsPresent + [int]$AtmosphereStudy.IsPresent + [int]$IntegrationStudy.IsPresent + [int]$TemporalStudy.IsPresent + [int]$DensityParity.IsPresent + [int]$DensityCacheStudy.IsPresent + [int]$Cached.IsPresent) -gt 1) { throw 'Choose only one test mode.' }
$projectRoot = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$shaderDirectory = Join-Path $projectRoot 'shaders'
$defines = @()
$cppDefines = @()
if ($Reference) {
    $shaderDirectory = Join-Path $OutputDirectory 'reference-source'
    New-Item -ItemType Directory -Force -Path $shaderDirectory | Out-Null
    Get-ChildItem -LiteralPath (Join-Path $projectRoot 'shaders') -File | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $shaderDirectory -Force }
    Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'cumulus_reference') -File | Where-Object { $_.Extension -in @('.h','.hlsl','.hlsli') } | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $shaderDirectory -Force }
    $defines = @('-D','CUMULUS_REFERENCE=1')
    $cppDefines = @('/DCUMULUS_REFERENCE=1')
}
$targets = @(
    @('noise','Pass_cumulus_noise_v8.hlsl','main'),
    @('light','Pass_cumulus_light_v8.hlsl','main'),
    @('trans','Pass_skylut_bake_v8.hlsl','mainTransmittance'),
    @('multi','Pass_skylut_bake_v8.hlsl','mainMultiScatter'),
    @('environment','Pass_cumulus_environment_v8.hlsl','main'),
    @('ambient','Pass_cumulus_ambient_v8.hlsl','main'),
    @('secondary','Pass_cumulus_secondary_v8.hlsl','main'),
    @('seedSecondary','../tests/CumulusGpuTests.hlsl','seedSecondary'),
    @('checkSecondary','../tests/CumulusGpuTests.hlsl','checkSecondary'),
    @('preview','../tests/CumulusGpuTests.hlsl','main'),
    @('material','../tests/CumulusGpuTests.hlsl','probeMaterial'))
if (-not $Reference) {
    $targets += ,@('densityCache','Pass_cumulus_density_v8.hlsl','main')
    $targets += ,@('densityCacheProbe','../tests/CumulusDensityCacheTests.hlsl','densityCacheProbe')
    $targets += ,@('integrationProbe','../tests/CumulusIntegrationTests.hlsl','integrationProbe')
    $targets += ,@('guides','../tests/CumulusGpuTests.hlsl','writeGuides')
    $targets += ,@('airQuadrature','../tests/CumulusIntegrationTests.hlsl','airQuadratureProbe')
    $targets += ,@('sunOcclusion','../tests/CumulusIntegrationTests.hlsl','sunOcclusionProbe')
    $targets += ,@('densityParity','../tests/CumulusDensityParity.hlsl','densityParity')
    $targets += ,@('densityReference','../tests/CumulusDensityParity.hlsl','densityReference')
}
foreach ($target in $targets) {
    $inputShader = if ($target[1].StartsWith('../tests/')) { Join-Path $projectRoot ("tests/" + (Split-Path $target[1] -Leaf)) } else { Join-Path $shaderDirectory $target[1] }
    # Windows PowerShell classifies native compiler warnings as error records.
    # Preserve their log but decide success from DXC's exit code.
    $ErrorActionPreference = 'Continue'
    & "$projectRoot/include/dxc.exe" -T cs_6_6 -E $target[2] -HV 2021 -enable-16bit-types -O3 `
        -I $shaderDirectory -I "$projectRoot/include" @defines `
        $inputShader -Fo "$OutputDirectory/$($target[0]).dxil" *> "$OutputDirectory/$($target[0]).log"
    $ErrorActionPreference = 'Stop'
    if ($LASTEXITCODE -ne 0) { Get-Content "$OutputDirectory/$($target[0]).log"; throw 'Cloud shader compilation failed' }
}
& cl.exe /nologo /std:c++17 /EHsc /O2 @cppDefines "$PSScriptRoot/CumulusGpuTests.cpp" `
    "/Fo$OutputDirectory/CumulusGpuTests.obj" "/Fe$OutputDirectory/CumulusGpuTests.exe" /link d3d12.lib dxgi.lib
if ($LASTEXITCODE -ne 0) { throw 'Cumulus GPU runner compilation failed; use a VS Developer shell' }
$testArgs = @($OutputDirectory,$Width,$Height)
if ($LightingStudy) {
    $testArgs += '--lighting'
}
if ($Benchmark) { $testArgs += '--benchmark' }
if ($Restored) { $testArgs += '--restored' }
if ($DetailStudy) { $testArgs += '--detail' }
if ($TwilightStudy) { $testArgs += '--twilight' }
if ($AtmosphereStudy) { $testArgs += '--atmosphere' }
if ($IntegrationStudy) { $testArgs += '--integration' }
if ($TemporalStudy) { $testArgs += '--temporal' }
if ($DensityParity) { $testArgs += '--density-parity' }
if ($DensityCacheStudy) { $testArgs += '--density-cache' }
if ($Cached) { $testArgs += '--cached' }
& "$OutputDirectory/CumulusGpuTests.exe" @testArgs
if ($LASTEXITCODE -ne 0) { throw 'Cumulus GPU checks failed' }
