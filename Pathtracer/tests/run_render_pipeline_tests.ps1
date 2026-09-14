param([string]$OutputDirectory = (Join-Path $PSScriptRoot '../out/render-pipeline-tests'))
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path $PSScriptRoot -Parent
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$OutputDirectory = (Resolve-Path $OutputDirectory).Path
& cl.exe /nologo /std:c++17 /EHsc /O2 /DNOMINMAX "/I$projectRoot/include" "/I$projectRoot/rdn" "/I$projectRoot/DirectXTex" `
    "$PSScriptRoot/RenderPipelineTests.cpp" "$projectRoot/rdn/Raytracing/PassSystem.cpp" `
    "/Fo$OutputDirectory/" "/Fe$OutputDirectory/RenderPipelineTests.exe"
if ($LASTEXITCODE -ne 0) { throw 'Pipeline test compilation failed. Use a VS Developer shell.' }
& "$OutputDirectory/RenderPipelineTests.exe"
if ($LASTEXITCODE -ne 0) { throw 'Pipeline regression failed' }
