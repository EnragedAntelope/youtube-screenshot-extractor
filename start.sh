#!/bin/bash

# YouTube Screenshot Extractor - Setup and Launch Script
# For macOS and Linux

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    else
        OS="unknown"
    fi
}

# Print colored message
print_message() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Print header
print_header() {
    clear
    echo ""
    print_message "$BLUE" "================================================================"
    print_message "$BLUE" "       YouTube Screenshot Extractor - Setup and Launch"
    print_message "$BLUE" "================================================================"
    echo ""
}

# Show main menu
show_menu() {
    print_header
    echo "  [1] Initial Setup - first time only"
    echo "      Creates virtual environment and installs dependencies"
    echo ""
    echo "  [2] Update yt-dlp"
    echo "      Keep yt-dlp current for YouTube compatibility"
    echo ""
    echo "  [3] Install Deno - REQUIRED for YouTube"
    echo "      JavaScript runtime needed for YouTube downloads"
    echo ""
    echo "  [4] Install FFmpeg - REQUIRED for keyframes/filters"
    echo "      Media processing tool needed for advanced features"
    echo ""
    echo "  [5] Install GPU Support - optional, rarely needed"
    echo "      PyCUDA for NVIDIA GPU - modern CPUs are usually fast enough"
    echo ""
    echo "  [6] Launch GUI"
    echo "      Start the graphical interface"
    echo ""
    echo "  [7] Launch Command Line Help"
    echo "      Show all command line options"
    echo ""
    echo "  [8] Exit"
    echo ""
    read -p "  Enter your choice [1-8]: " choice
    echo ""

    case $choice in
        1) initial_setup ;;
        2) update_ytdlp ;;
        3) install_deno ;;
        4) install_ffmpeg ;;
        5) install_gpu ;;
        6) launch_gui ;;
        7) launch_help ;;
        8) exit 0 ;;
        *)
            print_message "$RED" "  Invalid choice. Please try again."
            sleep 2
            show_menu
            ;;
    esac
}

# Initial setup
initial_setup() {
    print_header
    print_message "$BLUE" "  Initial Setup"
    print_message "$BLUE" "================================================================"
    echo ""

    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        print_message "$RED" "  ERROR: Python 3 is not installed."
        echo "  Please install Python 3.10+ from:"
        if [[ "$OS" == "macos" ]]; then
            echo "    - https://python.org OR"
            echo "    - brew install python3"
        else
            echo "    - https://python.org OR"
            echo "    - sudo apt install python3 python3-venv (Ubuntu/Debian)"
            echo "    - sudo dnf install python3 (Fedora)"
        fi
        echo ""
        read -p "Press Enter to continue..."
        show_menu
        return
    fi

    # Check Python version
    PYVER=$(python3 --version 2>&1 | awk '{print $2}')
    print_message "$GREEN" "  Found Python $PYVER"
    echo ""

    # Check if venv already exists
    if [ -d "venv" ]; then
        echo "  Virtual environment already exists."
        read -p "  Reinstall dependencies? [y/n]: " reinstall
        if [[ ! "$reinstall" =~ ^[Yy]$ ]]; then
            print_message "$YELLOW" "  Setup skipped."
            sleep 2
            show_menu
            return
        fi
    else
        print_message "$GREEN" "  Creating virtual environment..."
        python3 -m venv venv
        if [ $? -ne 0 ]; then
            print_message "$RED" "  ERROR: Failed to create virtual environment."
            read -p "Press Enter to continue..."
            show_menu
            return
        fi
        print_message "$GREEN" "  Virtual environment created."
        echo ""
    fi

    print_message "$GREEN" "  Activating virtual environment..."
    source venv/bin/activate

    print_message "$GREEN" "  Installing uv for faster package management..."
    pip install -q uv 2>/dev/null || true

    print_message "$GREEN" "  Installing dependencies - this may take a few minutes..."
    echo ""

    if command -v uv &> /dev/null; then
        uv pip install -r requirements.txt
    else
        pip install -r requirements.txt
    fi

    if [ $? -ne 0 ]; then
        print_message "$RED" "  ERROR: Failed to install some dependencies."
        echo "  Check the output above for details."
        read -p "Press Enter to continue..."
        show_menu
        return
    fi

    echo ""
    print_message "$GREEN" "================================================================"
    print_message "$GREEN" "  Setup Complete!"
    print_message "$GREEN" "================================================================"
    echo ""
    print_message "$YELLOW" "  IMPORTANT NEXT STEPS:"
    print_message "$YELLOW" "  - Run option [3] to install Deno (required for YouTube)"
    print_message "$YELLOW" "  - Run option [4] to install FFmpeg (required for keyframes/filters)"
    echo ""
    print_message "$GREEN" "  Then use option [6] to launch the GUI."
    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

# Update yt-dlp
update_ytdlp() {
    print_header
    print_message "$BLUE" "  Updating yt-dlp"
    print_message "$BLUE" "================================================================"
    echo ""

    if [ ! -d "venv" ]; then
        print_message "$RED" "  ERROR: Virtual environment not found."
        echo "  Please run Initial Setup first - option 1."
        echo ""
        read -p "Press Enter to continue..."
        show_menu
        return
    fi

    source venv/bin/activate
    print_message "$GREEN" "  Checking for yt-dlp updates..."
    echo ""

    if command -v uv &> /dev/null; then
        uv pip install --upgrade "yt-dlp[default]"
    else
        pip install --upgrade "yt-dlp[default]"
    fi

    echo ""
    print_message "$GREEN" "  yt-dlp has been updated to the latest version."
    echo ""
    print_message "$YELLOW" "  TIP: Run this regularly to maintain YouTube compatibility."
    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

# Install Deno
install_deno() {
    print_header
    print_message "$BLUE" "  Deno Installation - Required for YouTube"
    print_message "$BLUE" "================================================================"
    echo ""
    echo "  Deno is a JavaScript runtime REQUIRED for YouTube downloads."
    echo "  Without it, YouTube videos may fail to download."
    echo ""
    echo "  Other sites [1000+ supported] work without Deno."
    echo ""

    # Check if Deno is already installed
    if command -v deno &> /dev/null; then
        print_message "$GREEN" "  Deno is already installed!"
        deno --version
        echo ""
        read -p "Press Enter to continue..."
        show_menu
        return
    fi

    print_message "$YELLOW" "  Deno is NOT currently installed."
    echo ""

    if [[ "$OS" == "macos" ]]; then
        echo "  Installation options for macOS:"
        echo ""
        echo "  [1] Homebrew (recommended if you have brew)"
        echo "      brew install deno"
        echo ""
        echo "  [2] Official installer script (curl)"
        echo "      curl -fsSL https://deno.land/install.sh | sh"
        echo ""
        read -p "  Choose installation method [1/2] or 'n' to skip: " deno_choice

        case $deno_choice in
            1)
                if ! command -v brew &> /dev/null; then
                    print_message "$RED" "  Homebrew not found. Install from https://brew.sh"
                    read -p "Press Enter to continue..."
                    show_menu
                    return
                fi
                print_message "$GREEN" "  Installing Deno via Homebrew..."
                brew install deno
                ;;
            2)
                print_message "$GREEN" "  Installing Deno via official installer..."
                curl -fsSL https://deno.land/install.sh | sh
                echo ""
                print_message "$YELLOW" "  Add Deno to your PATH by adding this to ~/.zshrc or ~/.bash_profile:"
                print_message "$YELLOW" "  export PATH=\"\$HOME/.deno/bin:\$PATH\""
                ;;
            *)
                print_message "$YELLOW" "  Installation skipped."
                read -p "Press Enter to continue..."
                show_menu
                return
                ;;
        esac
    else
        echo "  Installation options for Linux:"
        echo ""
        echo "  [1] Official installer script (recommended)"
        echo "      curl -fsSL https://deno.land/install.sh | sh"
        echo ""
        echo "  [2] Package manager (varies by distribution)"
        echo "      - Arch: pacman -S deno"
        echo "      - Ubuntu/Debian: snap install deno"
        echo ""
        read -p "  Use official installer? [y/n]: " deno_choice

        if [[ "$deno_choice" =~ ^[Yy]$ ]]; then
            print_message "$GREEN" "  Installing Deno via official installer..."
            curl -fsSL https://deno.land/install.sh | sh
            echo ""
            print_message "$YELLOW" "  Add Deno to your PATH by adding this to ~/.bashrc or ~/.zshrc:"
            print_message "$YELLOW" "  export PATH=\"\$HOME/.deno/bin:\$PATH\""
        else
            print_message "$YELLOW" "  Please install manually using your package manager."
            echo "  Visit https://deno.land for more details."
        fi
    fi

    echo ""
    print_message "$YELLOW" "  NOTE: You may need to restart your terminal for Deno to be"
    print_message "$YELLOW" "  recognized in PATH. Run 'source ~/.bashrc' or 'source ~/.zshrc'"
    print_message "$YELLOW" "  (or close and reopen terminal)."
    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

# Install FFmpeg
install_ffmpeg() {
    print_header
    print_message "$BLUE" "  FFmpeg Installation - Required for Keyframes/Filters"
    print_message "$BLUE" "================================================================"
    echo ""
    echo "  FFmpeg is REQUIRED for:"
    echo "    - Keyframe extraction (--method keyframes)"
    echo "    - Post-processing filters (--gradfun, --deband)"
    echo "    - Audio merging during YouTube downloads"
    echo ""
    echo "  The tool will work without FFmpeg for basic interval extraction,"
    echo "  but most features require it."
    echo ""

    # Check if FFmpeg is already installed
    if command -v ffmpeg &> /dev/null; then
        print_message "$GREEN" "  FFmpeg is already installed!"
        ffmpeg -version | head -n 1
        echo ""
        read -p "Press Enter to continue..."
        show_menu
        return
    fi

    print_message "$YELLOW" "  FFmpeg is NOT currently installed."
    echo ""

    if [[ "$OS" == "macos" ]]; then
        echo "  Installation options for macOS:"
        echo ""
        echo "  [1] Homebrew (recommended)"
        echo "      brew install ffmpeg"
        echo ""
        echo "  [2] Manual download"
        echo "      https://ffmpeg.org/download.html"
        echo ""
        read -p "  Use Homebrew to install? [y/n]: " ffmpeg_choice

        if [[ "$ffmpeg_choice" =~ ^[Yy]$ ]]; then
            if ! command -v brew &> /dev/null; then
                print_message "$RED" "  Homebrew not found. Install from https://brew.sh"
                read -p "Press Enter to continue..."
                show_menu
                return
            fi
            print_message "$GREEN" "  Installing FFmpeg via Homebrew..."
            brew install ffmpeg
        else
            print_message "$YELLOW" "  Please download and install from https://ffmpeg.org/download.html"
        fi
    else
        echo "  Installation options for Linux:"
        echo ""
        echo "  [1] APT (Ubuntu/Debian)"
        echo "      sudo apt update && sudo apt install ffmpeg"
        echo ""
        echo "  [2] DNF (Fedora)"
        echo "      sudo dnf install ffmpeg"
        echo ""
        echo "  [3] Pacman (Arch)"
        echo "      sudo pacman -S ffmpeg"
        echo ""
        read -p "  Choose package manager [1/2/3] or 'n' to skip: " ffmpeg_choice

        case $ffmpeg_choice in
            1)
                print_message "$GREEN" "  Installing FFmpeg via APT..."
                sudo apt update && sudo apt install -y ffmpeg
                ;;
            2)
                print_message "$GREEN" "  Installing FFmpeg via DNF..."
                sudo dnf install -y ffmpeg
                ;;
            3)
                print_message "$GREEN" "  Installing FFmpeg via Pacman..."
                sudo pacman -S --noconfirm ffmpeg
                ;;
            *)
                print_message "$YELLOW" "  Installation skipped."
                echo "  Install manually using your package manager or from:"
                echo "  https://ffmpeg.org/download.html"
                ;;
        esac
    fi

    echo ""
    if command -v ffmpeg &> /dev/null; then
        print_message "$GREEN" "  FFmpeg installed successfully!"
    else
        print_message "$YELLOW" "  Please verify FFmpeg installation by running: ffmpeg -version"
    fi
    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

# Install GPU support
install_gpu() {
    print_header
    print_message "$BLUE" "  GPU Support Installation - Optional"
    print_message "$BLUE" "================================================================"
    echo ""
    echo "  This installs PyCUDA for NVIDIA GPU acceleration."
    echo ""
    print_message "$YELLOW" "  WARNING: This install takes 5-10 minutes and downloads large"
    print_message "$YELLOW" "  dependencies. Only recommended if processing many videos."
    echo ""
    echo "  REQUIREMENTS:"
    echo "    - NVIDIA GPU"
    echo "    - CUDA Toolkit - https://developer.nvidia.com/cuda-toolkit"
    if [[ "$OS" == "linux" ]]; then
        echo "    - build-essential (Ubuntu) or Development Tools (Fedora)"
    else
        echo "    - Xcode Command Line Tools (macOS)"
    fi
    echo ""
    echo "  Skip this if you don't have an NVIDIA GPU - the tool works"
    echo "  fine with CPU processing."
    echo ""
    read -p "  Proceed with installation? [y/n]: " confirm

    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        show_menu
        return
    fi

    if [ ! -d "venv" ]; then
        print_message "$RED" "  ERROR: Virtual environment not found."
        echo "  Please run Initial Setup first - option 1."
        echo ""
        read -p "Press Enter to continue..."
        show_menu
        return
    fi

    source venv/bin/activate
    echo ""
    print_message "$GREEN" "  Installing PyCUDA - this will take several minutes..."
    echo ""

    if command -v uv &> /dev/null; then
        uv pip install pycuda 2>/dev/null || pip install pycuda
    else
        pip install pycuda
    fi

    if [ $? -ne 0 ]; then
        echo ""
        print_message "$RED" "================================================================"
        print_message "$RED" "  PyCUDA installation failed."
        echo ""
        echo "  This is normal if you don't have:"
        echo "    - An NVIDIA GPU"
        echo "    - CUDA Toolkit installed"
        if [[ "$OS" == "linux" ]]; then
            echo "    - build-essential (sudo apt install build-essential)"
        else
            echo "    - Xcode Command Line Tools (xcode-select --install)"
        fi
        echo ""
        echo "  The tool will work fine without GPU acceleration."
        print_message "$RED" "================================================================"
    else
        echo ""
        print_message "$GREEN" "  PyCUDA installed successfully!"
        print_message "$GREEN" "  GPU acceleration is now available."
    fi

    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

# Launch GUI
launch_gui() {
    print_header
    print_message "$GREEN" "  Launching GUI..."
    echo ""

    if [ ! -d "venv" ]; then
        print_message "$RED" "  ERROR: Virtual environment not found."
        echo "  Please run Initial Setup first - option 1."
        echo ""
        read -p "Press Enter to continue..."
        show_menu
        return
    fi

    source venv/bin/activate

    # Launch GUI (python3 works on both macOS and Linux)
    python3 youtube-screenshot-gui.py &

    print_message "$GREEN" "  GUI launched in the background."
    echo ""
    sleep 2
    show_menu
}

# Launch help
launch_help() {
    print_header

    if [ ! -d "venv" ]; then
        print_message "$RED" "  ERROR: Virtual environment not found."
        echo "  Please run Initial Setup first - option 1."
        echo ""
        read -p "Press Enter to continue..."
        show_menu
        return
    fi

    source venv/bin/activate
    python youtube-screenshot-script.py --help
    echo ""
    print_message "$BLUE" "================================================================"
    print_message "$BLUE" "  Example Commands"
    print_message "$BLUE" "================================================================"
    echo ""
    echo "  Basic extraction:"
    echo "    python youtube-screenshot-script.py \"VIDEO_URL\""
    echo ""
    echo "  Optimal quality:"
    echo "    python youtube-screenshot-script.py \"VIDEO_URL\" \\"
    echo "      --quality 50 --blur 100 --detect-watermarks --deblock"
    echo ""
    echo "  Local file with scene detection:"
    echo "    python youtube-screenshot-script.py video.mp4 --method scene"
    echo ""
    read -p "Press Enter to continue..."
    show_menu
}

# Main script
detect_os

if [[ "$OS" == "unknown" ]]; then
    print_message "$RED" "Unsupported operating system: $OSTYPE"
    print_message "$RED" "This script supports macOS and Linux only."
    exit 1
fi

# Run menu loop
show_menu
