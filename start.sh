#!/bin/bash

# YouTube Screenshot Extractor - Setup and Launch Script
# For macOS and Linux

# Every path below (venv, requirements.txt, the Python scripts, the git
# checkout) is relative, so the script only works from the repo root. Move
# there rather than depending on the caller's working directory.
cd "$(dirname "${BASH_SOURCE[0]}")" || {
    echo "ERROR: could not change to the script's directory." >&2
    exit 1
}

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
    if [ -n "$VERDATE" ]; then
        print_message "$GREEN" "  Installed version: $VERDATE"
    fi
    print_message "$YELLOW" "  Tip: run [2] Check for Updates to stay current - especially"
    print_message "$YELLOW" "       if it's been a while since you last used this tool."
    echo ""
}

# Show main menu and dispatch a single choice. Returns to the main loop, which
# redraws the menu; each action returns rather than recursing into show_menu.
show_menu() {
    print_header
    echo "  [1] Launch GUI"
    echo "      Start the graphical interface"
    echo ""
    echo "  [2] Check for Updates"
    echo "      Pull the latest tool code and update yt-dlp"
    echo ""
    echo "  --- First-time setup ---"
    echo ""
    echo "  [3] Initial Setup - first time only"
    echo "      Creates virtual environment and installs dependencies"
    echo ""
    echo "  [4] Install Deno - REQUIRED for YouTube"
    echo "      JavaScript runtime needed for YouTube downloads"
    echo ""
    echo "  [5] Install FFmpeg - REQUIRED for keyframes/filters"
    echo "      Media processing tool needed for advanced features"
    echo ""
    echo "  [6] Launch Command Line Help"
    echo "      Show all command line options"
    echo ""
    echo "  [7] Exit"
    echo ""
    read -p "  Enter your choice [1-7]: " choice
    echo ""

    case $choice in
        1) launch_gui ;;
        2) update ;;
        3) initial_setup ;;
        4) install_deno ;;
        5) install_ffmpeg ;;
        6) launch_help ;;
        7) exit 0 ;;
        *)
            print_message "$RED" "  Invalid choice. Please try again."
            sleep 2
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
            return
        fi
    else
        print_message "$GREEN" "  Creating virtual environment..."
        python3 -m venv venv
        if [ $? -ne 0 ]; then
            print_message "$RED" "  ERROR: Failed to create virtual environment."
            read -p "Press Enter to continue..."
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
        return
    fi

    echo ""
    print_message "$GREEN" "================================================================"
    print_message "$GREEN" "  Setup Complete!"
    print_message "$GREEN" "================================================================"
    echo ""
    print_message "$YELLOW" "  IMPORTANT NEXT STEPS:"
    print_message "$YELLOW" "  - Run option [4] to install Deno (required for YouTube)"
    print_message "$YELLOW" "  - Run option [5] to install FFmpeg (required for keyframes/filters)"
    echo ""
    print_message "$GREEN" "  Then use option [1] to launch the GUI."
    echo ""
    read -p "Press Enter to continue..."
}

# Check for updates (tool code via git + yt-dlp)
update() {
    print_header
    print_message "$BLUE" "  Check for Updates"
    print_message "$BLUE" "================================================================"
    echo ""

    LAUNCHER_CHANGED=0

    # --- Update yt-dlp first (needs the virtual environment) ---
    if [ ! -d "venv" ]; then
        print_message "$YELLOW" "  Skipping yt-dlp update: run Initial Setup - option 3 - first."
    else
        source venv/bin/activate
        print_message "$GREEN" "  Checking for dependency updates..."
        echo ""
        # --upgrade on requirements.txt too, not just yt-dlp: pip leaves an
        # already-installed package alone when it still satisfies the floor, so
        # security bumps in requirements.txt never reach existing installs
        # otherwise.
        if command -v uv &> /dev/null; then
            uv pip install --upgrade -r requirements.txt
            uv pip install --upgrade "yt-dlp[default]"
        else
            pip install --upgrade -r requirements.txt
            pip install --upgrade "yt-dlp[default]"
        fi
        echo ""
        print_message "$GREEN" "  Dependencies are up to date."
    fi
    echo ""

    # --- Update the tool's own code via git ---
    if ! command -v git &> /dev/null; then
        print_message "$YELLOW" "  Git is not installed, so the tool's code cannot be auto-updated."
        echo "  Install Git, or download the latest version from:"
        echo "    https://github.com/EnragedAntelope/youtube-screenshot-extractor"
    elif [ ! -d ".git" ]; then
        print_message "$YELLOW" "  This folder is not a git checkout - likely downloaded as a ZIP -"
        echo "  so the tool's code cannot be auto-updated. To get updates automatically,"
        echo "  clone the repo instead:"
        echo "    git clone https://github.com/EnragedAntelope/youtube-screenshot-extractor.git"
    else
        print_message "$GREEN" "  Checking for tool updates..."
        OLDREV=$(git rev-parse HEAD 2>/dev/null)
        if git pull --ff-only; then
            NEWREV=$(git rev-parse HEAD 2>/dev/null)
            if [ "$OLDREV" == "$NEWREV" ]; then
                print_message "$GREEN" "  Tool code is already up to date."
            else
                print_message "$GREEN" "  Tool updated to the latest version."
                if git diff --name-only "$OLDREV" "$NEWREV" | grep -qi "start.sh"; then
                    LAUNCHER_CHANGED=1
                fi
            fi
        else
            print_message "$RED" "  Could not update automatically (local changes or network problem)."
            echo "  Your files were NOT modified. Resolve local changes, or update manually."
        fi
    fi

    echo ""
    if [ "$LAUNCHER_CHANGED" == "1" ]; then
        print_message "$YELLOW" "  ================================================================"
        print_message "$YELLOW" "  The launcher - start.sh - itself was updated."
        print_message "$YELLOW" "  Please re-run ./start.sh to load the new version."
        print_message "$YELLOW" "  ================================================================"
        echo ""
        read -p "Press Enter to exit..."
        exit 0
    fi
    print_message "$GREEN" "  All set. Updated code takes effect the next time you launch the GUI -"
    print_message "$GREEN" "  no restart of this menu needed."
    echo ""
    read -p "Press Enter to continue..."
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
}

# Launch GUI
launch_gui() {
    print_header
    print_message "$GREEN" "  Launching GUI..."
    echo ""

    if [ ! -d "venv" ]; then
        print_message "$RED" "  ERROR: Virtual environment not found."
        echo "  Please run Initial Setup first - option 3."
        echo ""
        read -p "Press Enter to continue..."
        return
    fi

    source venv/bin/activate

    # Launch GUI (python3 works on both macOS and Linux)
    python3 youtube-screenshot-gui.py &

    print_message "$GREEN" "  GUI launched in the background."
    echo ""
    sleep 2
}

# Launch help
launch_help() {
    print_header

    if [ ! -d "venv" ]; then
        print_message "$RED" "  ERROR: Virtual environment not found."
        echo "  Please run Initial Setup first - option 3."
        echo ""
        read -p "Press Enter to continue..."
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
    echo "  YouTube with authentication (for age-restricted/PO Token videos):"
    echo "    python youtube-screenshot-script.py \"URL\" --cookies-from-browser firefox"
    echo ""
    echo "  Avoid rate limiting (add delay between requests):"
    echo "    python youtube-screenshot-script.py \"URL\" --sleep-requests 5"
    echo ""
    read -p "Press Enter to continue..."
}

# Main script
detect_os

if [[ "$OS" == "unknown" ]]; then
    print_message "$RED" "Unsupported operating system: $OSTYPE"
    print_message "$RED" "This script supports macOS and Linux only."
    exit 1
fi

# Determine the installed version date (only if this is a git checkout)
VERDATE=""
if command -v git &> /dev/null && [ -d ".git" ]; then
    VERDATE=$(git log -1 --format=%cs 2>/dev/null)
fi

# Run menu loop - each selection returns here and the menu redraws
while true; do
    show_menu
done
