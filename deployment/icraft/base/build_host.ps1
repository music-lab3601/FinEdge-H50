param([string]$Python="python")
$repo=Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
& $Python (Join-Path $repo "finedge.py") model tools --stage base
exit $LASTEXITCODE
