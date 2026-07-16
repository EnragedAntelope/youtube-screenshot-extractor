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
echo   [4] Install FFmpeg - REQUIRED for keyframes/filters
echo       Media processing tool needed for advanced features
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
if "%choice%"=="4" goto install_ffmpeg
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
echo  IMPORTANT NEXT STEPS:
echo  - Run option [3] to install Deno (required for YouTube)
echo  - Run option [4] to install FFmpeg (required for keyframes/filters)
echo.
echo  Then use option [5] to launch the GUI.
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
uv pip install --upgrade "yt-dlp[default]" 2>nul || pip install --upgrade "yt-dlp[default]"
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

:install_ffmpeg
cls
echo.
echo  ================================================================
echo   FFmpeg Installation - Required for Keyframes/Filters
echo  ================================================================
echo.
echo  FFmpeg is REQUIRED for:
echo    - Keyframe extraction (--method keyframes)
echo    - Post-processing filters (--gradfun, --deband)
echo    - Audio merging during YouTube downloads
echo.
echo  The tool will work without FFmpeg for basic interval extraction,
echo  but most features require it.
echo.

REM Check if FFmpeg is already installed
ffmpeg -version >nul 2>&1
if not errorlevel 1 (
    echo  FFmpeg is already installed!
    ffmpeg -version 2^>^&1 | findstr "ffmpeg version"
    echo.
    pause
    goto menu
)

echo  FFmpeg is NOT currently installed.
echo.
echo  Installation options for Windows:
echo.
echo  [1] Chocolatey (recommended if you have choco)
echo      choco install ffmpeg
echo.
echo  [2] Winget
echo      winget install Gyan.FFmpeg
echo.
echo  [3] Manual download
echo      https://ffmpeg.org/download.html
echo.
set /p ffmpeg_choice="  Choose installation method [1/2/3] or 'n' to skip: "

if "%ffmpeg_choice%"=="1" (
    echo.
    echo  Checking for Chocolatey...
    choco --version >nul 2>&1
    if errorlevel 1 (
        echo.
        echo  Chocolatey not found. Install from https://chocolatey.org
        echo  Or choose a different installation method.
    ) else (
        echo.
        echo  Installing FFmpeg via Chocolatey...
        choco install ffmpeg -y
        if errorlevel 1 (
            echo.
            echo  Installation failed. Try manual installation.
        ) else (
            echo.
            echo  FFmpeg installed successfully!
            echo  NOTE: You may need to restart your terminal.
        )
    )
) else if "%ffmpeg_choice%"=="2" (
    echo.
    echo  Installing FFmpeg via winget...
    winget install --id=Gyan.FFmpeg -e
    if errorlevel 1 (
        echo.
        echo  Winget installation failed.
        echo  Try chocolatey or manual installation.
    ) else (
        echo.
        echo  FFmpeg installed successfully!
        echo  NOTE: You may need to restart your terminal.
    )
) else if "%ffmpeg_choice%"=="3" (
    echo.
    echo  Manual installation:
    echo    1. Go to https://www.gyan.dev/ffmpeg/builds/
    echo    2. Download ffmpeg-release-essentials.zip
    echo    3. Extract the archive
    echo    4. Add the bin folder to your system PATH
    echo.
    echo  Or download from: https://ffmpeg.org/download.html
) else (
    echo.
    echo  Installation skipped.
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
start "" venv\Scripts\pythonw.exe youtube-screenshot-gui.py
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
echo   YouTube with authentication (for age-restricted/PO Token videos):
echo     python youtube-screenshot-script.py "URL" --cookies-from-browser firefox
echo.
echo   Avoid rate limiting (add delay between requests):
echo     python youtube-screenshot-script.py "URL" --sleep-requests 5
echo.
pause
goto menu

:end
echo.
echo  Goodbye!
timeout /t 1 >nul
exit /b 0
