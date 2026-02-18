#Requires -Version 5.1
<#
.SYNOPSIS
    Digital Dharma v5.0 — Watchdog Startup Script
    Prevents Race Condition (TCP 10048) by enforcing single-watchdog and
    waiting for TIME_WAIT port release before each server restart.
#>

$WatchdogPidFile = "$PSScriptRoot\watchdog.pid"
$PidFile         = "$PSScriptRoot\server.pid"
$LogFile         = "$PSScriptRoot\dharma-server.log"
$Port            = 8000
$MaxRetries      = 10
$RetryDelaySec   = 5
$retryCount      = 0

function Write-Log {
    param([string]$Level, [string]$Message)
    $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$ts] [$Level] $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line -Encoding UTF8
}

# ===================================================================
# DUPLICATE WATCHDOG GUARD — 二重起動を即時解脱
# ===================================================================
if (Test-Path $WatchdogPidFile) {
    $staleWdPid = (Get-Content $WatchdogPidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
    if ($staleWdPid -and $staleWdPid.Trim()) {
        try {
            $wdProc = Get-Process -Id ([int]$staleWdPid.Trim()) -ErrorAction Stop
            if ($wdProc.ProcessName -match "pwsh|powershell") {
                Write-Log "INFO" "Watchdog PID $($staleWdPid.Trim()) is already running. Exiting (解脱)."
                exit 0
            }
        } catch {
            # PID が存在しない → stale ファイル。続行する
        }
    }
}
# 自身の PID を記録
$PID | Out-File -FilePath $WatchdogPidFile -Encoding ASCII -Force
Write-Log "INFO" "Watchdog started. PID=$PID"

# ===================================================================
# CLEANUP on exit (Ctrl+C / script end)
# ===================================================================
$cleanupBlock = {
    if (Test-Path $WatchdogPidFile) { Remove-Item $WatchdogPidFile -Force -ErrorAction SilentlyContinue }
    if (Test-Path $PidFile)         { Remove-Item $PidFile         -Force -ErrorAction SilentlyContinue }
    Write-Host "[INFO] Watchdog cleaned up and exited."
}
Register-EngineEvent -SourceIdentifier PowerShell.Exiting -Action $cleanupBlock | Out-Null

# ===================================================================
# SERVER RESTART LOOP
# ===================================================================
while ($retryCount -lt $MaxRetries) {

    # --- 古いサーバー PID を強制終了 ---
    if (Test-Path $PidFile) {
        $oldPid = (Get-Content $PidFile -ErrorAction SilentlyContinue | Select-Object -First 1)
        if ($oldPid -and $oldPid.Trim()) {
            try {
                Stop-Process -Id ([int]$oldPid.Trim()) -Force -ErrorAction Stop
                Write-Log "INFO" "Killed stale server PID $($oldPid.Trim())."
            } catch { <# already dead #> }
        }
        Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
    }

    # --- uvicorn 起動 ---
    Write-Log "INFO" "Starting uvicorn on port $Port (attempt $($retryCount + 1)/$MaxRetries)..."
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName  = "python"
    $psi.Arguments = "-m uvicorn server:app --host 0.0.0.0 --port $Port"
    $psi.WorkingDirectory = $PSScriptRoot
    $psi.UseShellExecute = $false

    $process = [System.Diagnostics.Process]::Start($psi)
    if (-not $process) {
        Write-Log "ERROR" "Failed to start process. Retrying..."
        $retryCount++
        Start-Sleep -Seconds $RetryDelaySec
        continue
    }

    $process.Id | Out-File -FilePath $PidFile -Encoding ASCII -Force
    Write-Log "INFO" "Server started. PID=$($process.Id)"

    # --- プロセス終了まで待機 ---
    $process.WaitForExit()
    $exitCode = $process.ExitCode
    Write-Log "WARN" "Server exited with code $exitCode."

    # --- クリーンアップ ---
    if (Test-Path $PidFile) { Remove-Item $PidFile -Force -ErrorAction SilentlyContinue }
    try { $process.Dispose() } catch {}

    # --- TIME_WAIT Meditation — ポートが完全解放されるまで待機 ---
    $waited = $false
    while (netstat -ano 2>$null | Select-String ":${Port}\s.*LISTENING") {
        if (-not $waited) {
            Write-Log "WARN" "Port $Port still bound (TIME_WAIT). Meditating..."
            $waited = $true
        }
        Start-Sleep -Seconds 5
    }
    if ($waited) { Write-Log "INFO" "Port $Port is now free. Resuming." }

    $retryCount++
    if ($retryCount -lt $MaxRetries) {
        Write-Log "INFO" "Restarting in ${RetryDelaySec}s... ($retryCount/$MaxRetries)"
        Start-Sleep -Seconds $RetryDelaySec
    }
}

Write-Log "ERROR" "Max retries ($MaxRetries) reached. Watchdog giving up."
if (Test-Path $WatchdogPidFile) { Remove-Item $WatchdogPidFile -Force -ErrorAction SilentlyContinue }
exit 1
