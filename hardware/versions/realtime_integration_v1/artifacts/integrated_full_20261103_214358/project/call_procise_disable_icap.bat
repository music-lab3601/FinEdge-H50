@set PROCISE_DIR=@PROCISE_ROOT@
@set TCL_LIBRARY=%PROCISE_DIR%/tcl8.4
@set PATH=%PROCISE_DIR%/dll;%PROCISE_DIR%/bin;%PATH%
@set ICTIME_HOME=%PROCISE_DIR%
@set APP_DIR=%PROCISE_DIR%
@set FMSH_DB=%PROCISE_DIR%/db
@procise.exe @HARDWARE_ROOT@/versions/realtime_integration_v1/artifacts/integrated_full_20261103_214358/project/procise_run_disable_icap.tcl
