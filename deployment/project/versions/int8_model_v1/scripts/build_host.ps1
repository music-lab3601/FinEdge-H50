$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$projectRoot = Split-Path -Parent (Split-Path -Parent $taskRoot)
$cmakeExe = Join-Path $taskRoot '.venv\Scripts\cmake.exe'
$toolchain = Join-Path $projectRoot '.tools\vs_buildtools'
& $cmakeExe -S (Join-Path $taskRoot 'cpp') -B (Join-Path $taskRoot 'build') -G 'Visual Studio 17 2022' -A x64 "-DCMAKE_GENERATOR_INSTANCE=$toolchain" '-DCMAKE_PREFIX_PATH=@ICRAFT_INSTALL@/CLI v3.36.1'
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $cmakeExe --build (Join-Path $taskRoot 'build') --config Release --parallel 1
exit $LASTEXITCODE
