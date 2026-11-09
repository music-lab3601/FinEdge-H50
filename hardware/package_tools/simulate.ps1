param([Parameter(Mandatory=$true)][string]$VivadoBin)
$ErrorActionPreference='Stop'
$packageRoot=Split-Path -Parent $PSScriptRoot
$projectRoot=Join-Path $packageRoot '01_工程源码\finedge_host_demo'
$rtl=Join-Path $projectRoot 'versions\realtime_integration_v1\artifacts\integrated_full_20261103_214358\project\rtl\bench'
$testRoot=Join-Path $projectRoot 'versions\window_stream_v1\tests'
$outputRoot=Join-Path $packageRoot ('local_build\simulation_'+(Get-Date -Format 'yyyyMMdd_HHmmss'))
New-Item -ItemType Directory -Path $outputRoot | Out-Null
$sources=@("$rtl\e_reset_sync.sv","$rtl\upstream\b1_mailbox.sv","$rtl\wt_async_fifo.sv","$rtl\wt_history.sv","$rtl\tt_native_packer.sv","$rtl\g_payload_bridge.sv")
Push-Location $outputRoot
try {
 foreach ($name in @('fifo','window_bridge','history','window_bridge_earliest')) {
  & (Join-Path $VivadoBin 'xvlog.bat') -sv @sources (Join-Path $testRoot "tb_$name.sv")
  if ($LASTEXITCODE -ne 0) {throw "xvlog failed: $name"}
  & (Join-Path $VivadoBin 'xelab.bat') -mt 2 "tb_$name" -s "review_$name"
  if ($LASTEXITCODE -ne 0) {throw "xelab failed: $name"}
  & (Join-Path $VivadoBin 'xsim.bat') "review_$name" -runall -log "$name.log"
  if ($LASTEXITCODE -ne 0) {throw "xsim failed: $name"}
  $expected=@{fifo='WT_FIFO_PASS';window_bridge='WT_BRIDGE_PASS';history='WT_HISTORY_PASS';window_bridge_earliest='WT_VERIFY_MCP_PASS response_delay=0'}[$name]
  if (-not (Select-String -LiteralPath "$name.log" -SimpleMatch $expected -Quiet)) {throw "Missing PASS marker: $name"}
 }
} finally {Pop-Location}
Write-Host "Synthetic RTL tests passed. Logs: $outputRoot"
