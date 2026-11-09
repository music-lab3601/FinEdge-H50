param([string]$Python="python")
$repo=Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
& $Python (Join-Path $repo "finedge.py") native
exit $LASTEXITCODE
