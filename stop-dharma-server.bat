@echo off
REM ============================================================
REM  Digital Dharma v5.0 — Server Shutdown Script
REM  Reads PID files written by Start-DharmaServer.ps1 and
REM  terminates server + watchdog processes.
REM ============================================================

set "SCRIPT_DIR=%~dp0"
set "PID_FILE=%SCRIPT_DIR%server.pid"
set "WD_PID_FILE=%SCRIPT_DIR%watchdog.pid"

REM --- Kill server process ---
if exist "%PID_FILE%" (
    set /p SERVER_PID=<"%PID_FILE%"
    echo [INFO] Stopping server PID %SERVER_PID% ...
    taskkill /PID %SERVER_PID% /F >nul 2>&1
    del "%PID_FILE%" >nul 2>&1
    echo [INFO] Server stopped.
) else (
    echo [WARN] No server.pid found. Server may not be running.
)

REM --- Kill watchdog process ---
if exist "%WD_PID_FILE%" (
    set /p WD_PID=<"%WD_PID_FILE%"
    echo [INFO] Stopping watchdog PID %WD_PID% ...
    taskkill /PID %WD_PID% /F >nul 2>&1
    del "%WD_PID_FILE%" >nul 2>&1
    echo [INFO] Watchdog stopped.
) else (
    echo [WARN] No watchdog.pid found.
)

echo [INFO] Dharma server shutdown complete.
