@set PROCISE_DIR=tool://procise/
@set TCL_LIBRARY=%PROCISE_DIR%/tcl8.4
@set PATH=%PROCISE_DIR%/dll;%PROCISE_DIR%/bin;%PATH%
@set ICTIME_HOME=%PROCISE_DIR%
@set APP_DIR=%PROCISE_DIR%
@set FMSH_DB=%PROCISE_DIR%/db
@procise.exe repo://hardware/scripts/procise_run_disable_icap.tcl
