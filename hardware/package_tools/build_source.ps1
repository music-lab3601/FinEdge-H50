param([string]$VsDevCmd='')
$ErrorActionPreference='Stop'
function Get-DemoSha256([string]$LiteralFile) {
 $stream=[IO.File]::OpenRead($LiteralFile)
 $hasher=[Security.Cryptography.SHA256]::Create()
 try { return ([BitConverter]::ToString($hasher.ComputeHash($stream))).Replace('-','').ToLowerInvariant() }
 finally { $stream.Dispose(); $hasher.Dispose() }
}

$packageRoot=Split-Path -Parent $PSScriptRoot
$runtimeRoot=Join-Path $packageRoot '01_工程源码\finedge_host_demo\versions\realtime_integration_v1\queued_v2\showcase_v5'
if (-not $VsDevCmd) {
 $vswhere=Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
 if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Install VS2022 C++ Build Tools, or pass -VsDevCmd.' }
 $vsRoot=& $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
 $VsDevCmd=Join-Path $vsRoot 'Common7\Tools\VsDevCmd.bat'
}
if (-not (Test-Path -LiteralPath $VsDevCmd)) { throw 'VsDevCmd.bat not found.' }
$outputRoot=Join-Path $runtimeRoot 'artifacts\native_source'
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$buildBat=Join-Path $outputRoot 'compile.cmd'
$nativeRoot=Join-Path $runtimeRoot 'native'
$text="@echo off`r`nchcp 65001 >nul`r`ncall `"$VsDevCmd`" -arch=x64 -host_arch=x64`r`nif errorlevel 1 exit /b 1`r`ncd /d `"$outputRoot`"`r`ncl /nologo /O2 /EHsc /std:c++17 /DWIN32_LEAN_AND_MEAN /DNOMINMAX /I `"$nativeRoot`" `"$nativeRoot\fair_source.cpp`" /Fe:fair_source.exe /link ws2_32.lib`r`nexit /b %errorlevel%`r`n"
[IO.File]::WriteAllText($buildBat,$text,[Text.UTF8Encoding]::new($false))
& $env:ComSpec /d /c "`"$buildBat`""
if ($LASTEXITCODE -ne 0) { throw 'Source build failed.' }
$executable=Join-Path $outputRoot 'fair_source.exe'
$sources=@{}
foreach ($name in @('fair_source.cpp','finedge_contract.h','platform.hpp')) {
 $file=Join-Path $nativeRoot $name
 $sources[$file]=(Get-DemoSha256 $file)
}
$manifest=@{status='passed';executable=$executable;sha256=(Get-DemoSha256 $executable);sources=$sources;TX_thread_file_IO=$false;measured_receiver_feedback=$false}
$json=$manifest|ConvertTo-Json -Depth 8
[IO.File]::WriteAllText((Join-Path $runtimeRoot 'artifacts\source_build.json'),$json,[Text.UTF8Encoding]::new($false))
Write-Host 'Built independent native market dispatcher. No stock records or board accessed.'
