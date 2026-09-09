param(
    [Parameter(Mandatory=$true)][string]$RuntimePath,
    [string]$OutputDirectory = (Join-Path $PSScriptRoot '../out/dlssnr-tests')
)
$ErrorActionPreference = 'Stop'
$root = Split-Path $PSScriptRoot -Parent
$RuntimePath = (Resolve-Path -LiteralPath $RuntimePath).Path
if ((Get-FileHash -LiteralPath $RuntimePath -Algorithm SHA256).Hash -ne 'E16BCF15E16E13F527491CDF7845B2FE6521A738D8F7C9C721866A8496E1FC8E') { throw 'Unrecognized NR runtime; see docs/DLSS5.md' }
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $install = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $install) { throw 'Visual Studio C++ build tools required' }
    $devcmd = Join-Path $install 'Common7/Tools/VsDevCmd.bat'
    cmd /s /c "`"$devcmd`" -arch=x64 -host_arch=x64 >nul && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$') { [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process') }
    }
}
New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
$output = (Resolve-Path -LiteralPath $OutputDirectory).Path
$sdk = Join-Path $root 'streamline'
Copy-Item -Path "$sdk/bin/x64/*.dll" -Destination $output
& cl.exe /nologo /EHsc /O2 /std:c++20 /DNOMINMAX /DPATHTRACER_DLSSNR_NATIVE=0 "/I$sdk/include" "/I$root/rdn" "/I$root/DirectXTex" `
    "$PSScriptRoot/DLSSNRGpuTests.cpp" "$root/rdn/PostProcess/DLSSNRManager.cpp" "/Fo$output/" "/Fe$output/DLSSNRDisabledTests.exe" `
    /link "$sdk/lib/x64/sl.interposer.lib" d3d12.lib dxgi.lib wintrust.lib version.lib
if ($LASTEXITCODE -ne 0) { throw 'Disabled backend build failed' }
& "$output/DLSSNRDisabledTests.exe"
if ($LASTEXITCODE -ne 0) { throw 'Disabled backend test failed' }
& cl.exe /nologo /EHsc /O2 /std:c++17 /LD /DDLSSNR_BRIDGE_BUILD "/I$root/third_party/dlss-ngx/include" `
    "$root/rdn/PostProcess/DLSSNRBridge.cpp" "/Fo$output/bridge.obj" "/Fe$output/nvngx.dll_royaltracer_nr.dll" `
    /link bcrypt.lib "/IMPLIB:$output/bridge.lib"
if ($LASTEXITCODE -ne 0) { throw 'Bridge build failed' }
& "$root/include/dxc.exe" "$PSScriptRoot/DLSSNRIntegration.hlsl" -T cs_6_0 -E main -O3 -Fo "$output/DLSSNRIntegration.dxil"
if ($LASTEXITCODE -ne 0) { throw 'RR-to-NR test shader build failed' }
& cl.exe /nologo /EHsc /O2 /std:c++20 /DNOMINMAX /DPATHTRACER_DLSSNR_NATIVE=1 "/I$sdk/include" "/I$root/rdn" "/I$root/DirectXTex" `
    "$PSScriptRoot/DLSSNRGpuTests.cpp" "$root/rdn/PostProcess/DLSSNRManager.cpp" "$root/rdn/PostProcess/DLSSManager.cpp" "/Fo$output/" "/Fe$output/DLSSNRGpuTests.exe" `
    /link "$output/bridge.lib" "$sdk/lib/x64/sl.interposer.lib" d3d12.lib dxgi.lib wintrust.lib version.lib
if ($LASTEXITCODE -ne 0) { throw 'NR integration test build failed' }
Push-Location $output
try {
    & ./DLSSNRGpuTests.exe $RuntimePath
    if ($LASTEXITCODE -ne 0) { throw 'NR integration tests failed' }
} finally { Pop-Location }
