param([string]$Python="python",[string]$VivadoBin="")
$ErrorActionPreference="Stop"
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if ($VivadoBin) { & $Python (Join-Path $repo "finedge.py") configure --vivado (Join-Path $VivadoBin "vivado.bat"); if ($LASTEXITCODE) {exit $LASTEXITCODE} }
& $Python (Join-Path $repo "finedge.py") simulate
exit $LASTEXITCODE
