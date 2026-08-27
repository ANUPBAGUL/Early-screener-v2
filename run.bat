@echo off
setlocal enabledelayedexpansion

:: =====================================================================
::               EARLY SCREENER APP CONTROL SCRIPT
:: =====================================================================
:: Description: Start, stop, restart, and monitor the Streamlit application.
:: Usage: Run directly for interactive menu, or pass arguments:
::        run.bat start | stop | restart | status | setup
:: =====================================================================

:: Configuration variables
set "WINDOW_TITLE=EarlyScreenerApp"
set "PORT=8501"
set "SCRIPT_NAME=app.py"

:: Normalize project root directory path (stripping trailing backslash if present)
set "PROJECT_DIR=%~dp0"
if "%PROJECT_DIR:~-1%"=="\" set "PROJECT_DIR=%PROJECT_DIR:~0,-1%"

set "SCRIPT_PATH=%PROJECT_DIR%\%SCRIPT_NAME%"
set "REQUIREMENTS_PATH=%PROJECT_DIR%\requirements.txt"

:: Check command line arguments
if "%~1"=="start" goto :start_app
if "%~1"=="stop" goto :stop_app
if "%~1"=="restart" goto :restart_app
if "%~1"=="status" goto :status_app
if "%~1"=="setup" goto :setup_env
if not "%~1"=="" (
    echo [ERROR] Unknown parameter: %1
    echo Usage: %~nx0 [start^|stop^|restart^|status^|setup]
    exit /b 1
)

:menu
cls
echo =====================================================================
echo    🚀 EARLY SCREENER - APPLICATION MANAGER
echo =====================================================================
echo.
echo  [1] Start Application (New Window)
echo  [2] Start Application (Foreground / This Window)
echo  [3] Stop Application (Shutdown Server)
echo  [4] Restart Application
echo  [5] Check Server Status
echo  [6] Setup Environment (Install/Update Dependencies)
echo  [7] Exit
echo.
echo =====================================================================
set /p choice="Enter your choice (1-7): "

if "%choice%"=="1" goto :start_app
if "%choice%"=="2" goto :start_foreground
if "%choice%"=="3" goto :stop_app
if "%choice%"=="4" goto :restart_app
if "%choice%"=="5" goto :status_app
if "%choice%"=="6" goto :setup_env
if "%choice%"=="7" exit /b 0

echo.
echo [!] Invalid choice. Please select a number between 1 and 7.
timeout /t 2 >nul 2>&1
goto :menu


:: ---------------------------------------------------------------------
:: Detect Virtual Environment (.venv or venv)
:: ---------------------------------------------------------------------
:detect_venv
set "VENV_ACTIVATE="
if exist "%PROJECT_DIR%\.venv\Scripts\activate.bat" (
    set "VENV_ACTIVATE=%PROJECT_DIR%\.venv\Scripts\activate.bat"
) else if exist "%PROJECT_DIR%\venv\Scripts\activate.bat" (
    set "VENV_ACTIVATE=%PROJECT_DIR%\venv\Scripts\activate.bat"
)
exit /b 0


:: ---------------------------------------------------------------------
:: Start Application (Background / New Window)
:: ---------------------------------------------------------------------
:start_app
call :detect_venv
echo.
echo [+] Starting Early Screener in a new window...

:: Check if already running
call :is_running >nul 2>&1
if !ERRORLEVEL! equ 1 (
    echo [!] Application is already running. Use stop or restart first.
    if "%~1"=="" (
        pause
        goto :menu
    )
    exit /b 0
)

if not "%VENV_ACTIVATE%"=="" (
    echo [+] Activating virtual environment: %VENV_ACTIVATE%
    start "%WINDOW_TITLE%" cmd /k "title %WINDOW_TITLE% && cd /d "%PROJECT_DIR%" && call "%VENV_ACTIVATE%" && python -m streamlit run "%SCRIPT_PATH%""
) else (
    echo [!] No virtual environment found. Running with global python.
    start "%WINDOW_TITLE%" cmd /k "title %WINDOW_TITLE% && cd /d "%PROJECT_DIR%" && python -m streamlit run "%SCRIPT_PATH%""
)

echo [✓] Start command sent. The application should open in your browser shortly.
timeout /t 3 >nul 2>&1
if "%~1"=="" goto :menu
exit /b 0


:: ---------------------------------------------------------------------
:: Start Application (Foreground / Current Window)
:: ---------------------------------------------------------------------
:start_foreground
call :detect_venv
echo.
echo [+] Starting Early Screener in the foreground.
echo [!] Press Ctrl+C in this terminal window to stop the application.
echo.

if not "%VENV_ACTIVATE%"=="" (
    echo [+] Activating virtual environment: %VENV_ACTIVATE%
    call "%VENV_ACTIVATE%"
)

cd /d "%PROJECT_DIR%"
python -m streamlit run "%SCRIPT_PATH%"
if "%~1"=="" goto :menu
exit /b 0


:: ---------------------------------------------------------------------
:: Stop Application
:: ---------------------------------------------------------------------
:stop_app
echo.
echo [-] Shutting down Early Screener...
set "STOPPED=0"

:: 1. Terminate window by title
tasklist /V /FI "IMAGENAME eq cmd.exe" | findstr "%WINDOW_TITLE%" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    taskkill /FI "WINDOWTITLE eq %WINDOW_TITLE%*" /T /F >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        echo [✓] Closed application window "%WINDOW_TITLE%".
        set "STOPPED=1"
    )
)

:: 2. Find and kill process listening on Port 8501
for /f "tokens=5" %%a in ('netstat -aon ^| findstr :%PORT% ^| findstr LISTENING') do (
    taskkill /F /PID %%a >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        echo [✓] Stopped process %%a listening on port %PORT%.
        set "STOPPED=1"
    )
)

:: 3. Fallback: Search using powershell for any streamlit process running our script
powershell -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*streamlit*app.py*' -and $_.CommandLine -notlike '*Get-CimInstance*' }; if ($p) { $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; exit 1 } else { exit 0 }" >nul 2>&1
if !ERRORLEVEL! equ 1 (
    echo [✓] Stopped orphaned streamlit processes.
    set "STOPPED=1"
)

if "%STOPPED%"=="1" (
    echo [✓] Application stopped successfully.
) else (
    echo [!] No running instances of Early Screener were found.
)

timeout /t 2 >nul 2>&1
if "%~1"=="" goto :menu
exit /b 0


:: ---------------------------------------------------------------------
:: Restart Application
:: ---------------------------------------------------------------------
:restart_app
call :stop_app
timeout /t 1 >nul 2>&1
call :start_app
if "%~1"=="" goto :menu
exit /b 0


:: ---------------------------------------------------------------------
:: Check Application Status
:: ---------------------------------------------------------------------
:status_app
echo.
echo [*] Checking server status...
call :is_running
if !ERRORLEVEL! equ 1 (
    echo [STATUS] Running.
) else (
    echo [STATUS] Stopped.
)
echo.
pause
if "%~1"=="" goto :menu
exit /b 0


:: ---------------------------------------------------------------------
:: Helper to check if app is running (Exit code: 1 = Running, 0 = Stopped)
:: ---------------------------------------------------------------------
:is_running
set "RUNNING=0"

:: Check by Window Title
tasklist /V /FI "IMAGENAME eq cmd.exe" | findstr "%WINDOW_TITLE%" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set "RUNNING=1"
    echo [+] Found cmd window with title "%WINDOW_TITLE%".
)

:: Check by Port
netstat -aon | findstr :%PORT% | findstr LISTENING >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set "RUNNING=1"
    echo [+] Found process listening on port %PORT%.
)

:: Check by Process Command Line
powershell -Command "$p = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*streamlit*app.py*' -and $_.CommandLine -notlike '*Get-CimInstance*' }; if ($p) { exit 1 } else { exit 0 }" >nul 2>&1
if !ERRORLEVEL! equ 1 (
    set "RUNNING=1"
    echo [+] Found active streamlit python process running %SCRIPT_NAME%.
)

if "%RUNNING%"=="1" (
    exit /b 1
) else (
    exit /b 0
)


:: ---------------------------------------------------------------------
:: Environment Setup (Virtual Environment & pip dependencies)
:: ---------------------------------------------------------------------
:setup_env
echo.
echo [*] Setting up application environment...

:: Check if Python is installed
python --version >nul 2>&1
if !ERRORLEVEL! neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please download and install Python 3.8+ from https://www.python.org/
    pause
    if "%~1"=="" goto :menu
    exit /b 1
)

call :detect_venv
if not "%VENV_ACTIVATE%"=="" (
    echo [+] Activating existing virtual environment...
    call "%VENV_ACTIVATE%"
) else (
    echo [!] No virtual environment found.
    set /p make_venv="Would you like to create a new virtual environment (.venv)? (y/n): "
    if /i "!make_venv!"=="y" (
        echo [+] Creating virtual environment in %PROJECT_DIR%\.venv...
        python -m venv "%PROJECT_DIR%\.venv"
        if exist "%PROJECT_DIR%\.venv\Scripts\activate.bat" (
            set "VENV_ACTIVATE=%PROJECT_DIR%\.venv\Scripts\activate.bat"
            call "!VENV_ACTIVATE!"
        ) else (
            echo [ERROR] Failed to create virtual environment. Continuing with system Python.
        )
    )
)

if exist "%REQUIREMENTS_PATH%" (
    echo [+] Installing/Updating dependencies from requirements.txt...
    python -m pip install --upgrade pip
    pip install -r "%REQUIREMENTS_PATH%"
    if !ERRORLEVEL! equ 0 (
        echo [✓] Setup completed successfully!
    ) else (
        echo [ERROR] Failed to install some dependencies. Please check the logs.
    )
) else (
    echo [WARNING] requirements.txt not found. Cannot install dependencies automatically.
)

pause
if "%~1"=="" goto :menu
exit /b 0
