@echo off
setlocal enabledelayedexpansion
title YouTube Screenshot Extractor
color 0A

:menu
cls
echo.
echo  ================================================================
echo            YouTube Screenshot Extractor - Setup and Launch
echo  ================================================================
echo.
echo   [1] Initial Setup - first time only
echo       Creates virtual environment and installs dependencies
echo.
echo   [2] Update yt-dlp
echo       Keep yt-dlp current for YouTube compatibility
echo.
echo   [3] Install Deno - REQUIRED for YouTube
echo       JavaScript runtime needed for YouTube downloads
echo.
echo   [4] Install GPU Support - optional, rarely needed
echo       PyCUDA for NVIDIA GPU - modern CPUs are usually fast enough
echo.
echo   [5] Launch GUI
echo       Start the graphical interface
echo.
echo   [6] Launch Command Line Help
echo       Show all command line options
echo.
echo   [7] Exit
echo.
set /p choice="  Enter your choice [1-7]: "

if "%choice%"=="1" goto setup
if "%choice%"=="2" goto update_ytdlp
if "%choice%"=="3" goto install_deno
if "%choice%"=="4" goto install_gpu
if "%choice%"=="5" goto launch_gui
if "%choice%"=="6" goto launch_help
if "%choice%"=="7" goto end

echo.
echo  Invalid choice. Please try again.
timeout /t 2 >nul
goto menu

:setup
cls
echo.
echo  ================================================================
echo   Initial Setup
echo  ================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo  ERROR: Python is not installed or not in PATH.
    echo  Please install Python 3.10+ from https://python.org
    echo.
    pause
    goto menu
)

REM Check Python version
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo  Found Python %PYVER%
echo.

REM Check if venv already exists
if exist "venv" (
    echo  Virtual environment already exists.
    set /p reinstall="  Reinstall dependencies? [y/n]: "
    if /i "!reinstall!"=="y" (
        goto install_deps
    ) else (
        echo  Setup skipped.
        timeout /t 2 >nul
        goto menu
    )
)

echo  Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo  ERROR: Failed to create virtual environment.
    pause
    goto menu
)
echo  Virtual environment created.
echo.

:install_deps
echo  Activating virtual environment...
call venv\Scripts\activate.bat

echo  Installing uv for faster package management...
pip install uv >nul 2>&1

echo  Installing dependencies - this may take a few minutes...
echo.
uv pip install -r requirements.txt
if errorlevel 1 (
    echo  uv failed, falling back to pip...
    pip install -r requirements.txt
)
if errorlevel 1 (
    echo.
    echo  ERROR: Failed to install some dependencies.
    echo  Check the output above for details.
    pause
    goto menu
)

echo.
echo  ================================================================
echo   Setup Complete!
echo  ================================================================
echo.
echo  IMPORTANT: For YouTube downloads, you need Deno installed.
echo  Run option [3] to install Deno if you haven't already.
echo.
echo  Use option [5] to launch the GUI.
echo.
pause
goto menu

:update_ytdlp
cls
echo.
echo  ================================================================
echo   Updating yt-dlp
echo  ================================================================
echo.

if not exist "venv" (
    echo  ERROR: Virtual environment not found.
    echo  Please run Initial Setup first - option 1.
    echo.
    pause
    goto menu
)

call venv\Scripts\activate.bat
echo  Checking for yt-dlp updates...
echo.
uv pip install --upgrade yt-dlp 2>nul || pip install --upgrade yt-dlp
echo.
echo  yt-dlp has been updated to the latest version.
echo.
echo  TIP: Run this regularly to maintain YouTube compatibility.
echo.
pause
goto menu

:install_deno
cls
echo.
echo  ================================================================
echo   Deno Installation - Required for YouTube
echo  ================================================================
echo.
echo  Deno is a JavaScript runtime REQUIRED for YouTube downloads.
echo  Without it, YouTube videos may fail to download.
echo.
echo  Other sites [1000+ supported] work without Deno.
echo.

REM Check if Deno is already installed
deno --version >nul 2>&1
if not errorlevel 1 (
    echo  Deno is already installed!
    deno --version
    echo.
    pause
    goto menu
)

echo  Deno is NOT currently installed.
echo.
set /p deno_choice="  Install Deno automatically using winget? [y/n]: "
if /i not "%deno_choice%"=="y" (
    echo.
    echo  To install manually:
    echo    - Go to https://deno.land
    echo    - Download and run the Windows installer
    echo    - Restart this menu when done
    echo.
    pause
    goto menu
)

echo.
echo  Installing Deno via winget...
echo.
winget install --id=DenoLand.Deno -e
if errorlevel 1 (
    echo.
    echo  ================================================================
    echo  Winget installation failed.
    echo.
    echo  Try manual installation:
    echo    1. Go to https://deno.land
    echo    2. Download the Windows installer
    echo    3. Run the installer
    echo  ================================================================
) else (
    echo.
    echo  Deno installed successfully!
    echo.
    echo  NOTE: You may need to restart your terminal or computer
    echo  for Deno to be recognized in PATH.
)
echo.
pause
goto menu

:install_gpu
cls
echo.
echo  ================================================================
echo   GPU Support Installation - Optional
echo  ================================================================
echo.
echo  This installs PyCUDA for NVIDIA GPU acceleration.
echo.
echo  WARNING: This install takes 5-10 minutes and downloads large
echo  dependencies. Only recommended if processing many videos.
echo.
echo  REQUIREMENTS:
echo    - NVIDIA GPU
echo    - CUDA Toolkit - https://developer.nvidia.com/cuda-toolkit
echo    - Visual Studio Build Tools
echo.
echo  Skip this if you don't have an NVIDIA GPU - the tool works
echo  fine with CPU processing.
echo.
set /p confirm="  Proceed with installation? [y/n]: "
if /i not "%confirm%"=="y" goto menu

if not exist "venv" (
    echo.
    echo  ERROR: Virtual environment not found.
    echo  Please run Initial Setup first - option 1.
    echo.
    pause
    goto menu
)

call venv\Scripts\activate.bat
echo.
echo  Installing PyCUDA - this will take several minutes...
echo.
uv pip install pycuda 2>nul || pip install pycuda
if errorlevel 1 (
    echo.
    echo  ================================================================
    echo  PyCUDA installation failed.
    echo.
    echo  This is normal if you don't have:
    echo    - An NVIDIA GPU
    echo    - CUDA Toolkit installed
    echo    - Visual Studio Build Tools
    echo.
    echo  The tool will work fine without GPU acceleration.
    echo  ================================================================
) else (
    echo.
    echo  PyCUDA installed successfully!
    echo  GPU acceleration is now available.
)
echo.
pause
goto menu

:launch_gui
cls
echo.
echo  Launching GUI...
echo.

if not exist "venv" (
    echo  ERROR: Virtual environment not found.
    echo  Please run Initial Setup first - option 1.
    echo.
    pause
    goto menu
)

call venv\Scripts\activate.bat
start "" pythonw youtube-screenshot-gui.py
echo  GUI launched in a new window.
echo.
timeout /t 2 >nul
goto menu

:launch_help
cls
echo.

if not exist "venv" (
    echo  ERROR: Virtual environment not found.
    echo  Please run Initial Setup first - option 1.
    echo.
    pause
    goto menu
)

call venv\Scripts\activate.bat
python youtube-screenshot-script.py --help
echo.
echo  ================================================================
echo   Example Commands
echo  ================================================================
echo.
echo   Basic extraction:
echo     python youtube-screenshot-script.py "VIDEO_URL"
echo.
echo   Optimal quality:
echo     python youtube-screenshot-script.py "VIDEO_URL"
echo       --quality 50 --blur 100 --detect-watermarks --deblock
echo.
echo   Local file with scene detection:
echo     python youtube-screenshot-script.py video.mp4 --method scene
echo.
pause
goto menu

:end
echo.
echo  Goodbye!
timeout /t 1 >nul
exit /b 0
