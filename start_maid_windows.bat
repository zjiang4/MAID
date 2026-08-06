@echo off
setlocal EnableExtensions
title MAID Windows Launcher

cd /d "%~dp0"
set "PYTHON_CMD=python"
set "PYPI_INDEX=https://pypi.org/simple"

echo [MAID] Checking the default Python environment...
"%PYTHON_CMD%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python was not found in PATH.
    echo Install Python 3.11 or newer from https://www.python.org/ and enable "Add Python to PATH".
    pause
    exit /b 1
)

"%PYTHON_CMD%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [MAID] pip is missing. Trying to enable it...
    "%PYTHON_CMD%" -m ensurepip --upgrade
    if errorlevel 1 (
        echo [ERROR] Unable to initialize pip.
        pause
        exit /b 1
    )
)

echo [MAID] Checking required packages...
"%PYTHON_CMD%" -c "import nicegui, fastapi, pandas, numpy, sklearn, openai, langchain_nvidia_ai_endpoints" >nul 2>&1
if errorlevel 1 (
    echo [MAID] Installing missing packages from the official PyPI index...
    "%PYTHON_CMD%" -m pip install --upgrade pip --index-url "%PYPI_INDEX%"
    if errorlevel 1 goto :install_failed
    "%PYTHON_CMD%" -m pip install -r requirements.txt --index-url "%PYPI_INDEX%"
    if errorlevel 1 goto :install_failed
) else (
    echo [MAID] Required packages are already installed.
)

echo [MAID] Starting the server in the background...
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$root=(Get-Location).Path; $url='http://127.0.0.1:8080'; $listener=Get-NetTCPConnection -LocalPort 8080 -State Listen -ErrorAction SilentlyContinue; if(-not $listener){$log=Join-Path $root 'maid_server.log'; $err=Join-Path $root 'maid_server_error.log'; $pidFile=Join-Path $root 'maid_server.pid'; $process=Start-Process -FilePath 'python' -ArgumentList 'multi_agent_dev.py' -WorkingDirectory $root -RedirectStandardOutput $log -RedirectStandardError $err -WindowStyle Hidden -PassThru; Set-Content -LiteralPath $pidFile -Value $process.Id -Encoding ascii}; for($i=0;$i -lt 120;$i++){try{$response=Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop; if($response.StatusCode -eq 200){Start-Process 'http://127.0.0.1:8080'; exit 0}}catch{}; Start-Sleep -Seconds 1}; Write-Host '[ERROR] MAID did not become ready within 120 seconds.' -ForegroundColor Red; Write-Host 'Check maid_server_error.log for details.'; exit 1"

if errorlevel 1 (
    pause
    exit /b 1
)

echo [MAID] UI opened at http://127.0.0.1:8080
echo [MAID] Server logs: maid_server.log and maid_server_error.log
powershell -NoProfile -Command "Start-Sleep -Seconds 3"
exit /b 0

:install_failed
echo [ERROR] Dependency installation failed.
echo Check your network connection and run this launcher again.
pause
exit /b 1
