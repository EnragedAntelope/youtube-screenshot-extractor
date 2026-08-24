# AGENTS.md — youtube-screenshot-extractor

Pull clean, high-quality still frames from videos — YouTube, 1000+ other sites via yt-dlp, or local files. Downloads (if needed), finds best frames, filters blurry/low-quality ones, crops black bars, saves the rest. Built for gathering training images for LoRAs/checkpoints. GUI + CLI, Python 3.10+, requires Deno (YouTube) and FFmpeg (keyframes).

**Deep reference: Previous CLAUDE.md contained implementation notes about YouTube PO Tokens, rate limiting, and authentication — that content is now archived in the project history**

## Current state

_Last verified: 2026-08-23_

- **Status:** working and maintained, no version number and no release tags — `git log` is the only version record. The recent history is a run of audits fixing silently-broken options rather than new features.
- **Works:** all four extraction methods (interval, every frame, keyframes, scene detection); automatic blur/quality filtering, black-bar cropping and watermark flagging; YouTube authentication via browser cookies plus request-rate throttling; resume for large extractions; parallel worker-pool processing that streams frames instead of loading the whole video; GUI and CLI at parity.
- **In progress:** nothing — recent history is two audit rounds: dependency/security floors plus the first test suite and CI, then a robustness/parity pass (Stop kills the whole process tree, failed downloads clean up their partial files, `--png` honored by keyframes, FFmpeg filter frames piped instead of temp-filed, GUI blur range matched to the CLI).
- **Known gaps / next steps:** tests cover the pure helpers and `process_frame` (`tests/`, run with `pytest`) — download and the whole GUI are still verified by hand, and tkinter is not installable in every sandbox so GUI changes need a real desktop; **YouTube extraction is inherently fragile** — yt-dlp must be kept current (launcher option 2, or `pip install --upgrade "yt-dlp[default]"`), and the working client selection changes over time; Deno is required for YouTube and FFmpeg for keyframes, so a partial install silently limits which methods work; the rate-limit and client-selection notes in *Conventions & gotchas* are the most perishable content in this file — re-verify them before trusting them.
- **Deep docs:** none — `README.md` is the user-facing reference. Earlier implementation notes on PO tokens and authentication live only in the git history.

## Architecture in 60 seconds

- **Any source:** YouTube + 1000+ sites via yt-dlp, or local video files
- **Four extraction methods:** interval (every N seconds), every frame, keyframes only, scene-change detection
- **Automatic quality control:** sharpness/blur and quality scoring filter out bad frames before saving
- **Black bar removal:** letterboxing and pillarboxing cropped automatically
- **Watermark detection:** flags likely-watermarked frames in filename
- **YouTube authentication:** browser cookie support (`--cookies-from-browser firefox/chrome/edge/safari`) for age-restricted and private videos, plus rate limiting (`--sleep-requests`)
- **Resume support:** pick large extractions up where they left off
- **Parallel processing:** streams frames through a worker pool instead of loading whole video into memory
- **Dual interface:** GUI (`youtube-screenshot-gui.py`) and CLI (`youtube-screenshot-script.py`)

## Layout

| File | Purpose |
|------|---------|
| `youtube-screenshot-script.py` | CLI for frame extraction |
| `youtube-screenshot-gui.py` | GUI (point-and-click interface) |
| `START.bat` / `start.sh` | Startup menu: setup, update, launch GUI, help |
| `requirements.txt` | Python dependencies (yt-dlp, etc.) |
| `requirements-dev.txt` | Test/lint dependencies (pytest, pyflakes) |
| `assets/` | Screenshots and documentation images |

## Build / test / run

```bash
# Run the tests (no Deno/FFmpeg needed for the unit tests)
pip install -r requirements-dev.txt
python -m pytest tests -v

# Windows quick start
START.bat

# Or manual setup
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

# Install Deno (required for YouTube)
winget install DenoLand.Deno   # Windows
brew install deno              # macOS
curl -fsSL https://deno.land/install.sh | sh  # Linux

# Install FFmpeg (required for keyframe extraction)
winget install Gyan.FFmpeg     # Windows
brew install ffmpeg            # macOS
sudo apt install ffmpeg        # Linux

# Launch GUI
python youtube-screenshot-gui.py

# CLI usage examples
python youtube-screenshot-script.py "https://youtube.com/watch?v=..." --method interval --interval 5
python youtube-screenshot-script.py "https://youtube.com/watch?v=..." --method keyframes
python youtube-screenshot-script.py "video.mp4" --method scene --cookies-from-browser firefox
python youtube-screenshot-script.py --help
```

## Conventions & gotchas

- **Deno required for YouTube** (other sites don't need it). Startup scripts handle installation.
- **FFmpeg required** for keyframe extraction, gradfun/deband filters, merging separate audio/video streams. Interval/scene/all extraction and deblock filter work without it.
- **Keep yt-dlp updated** — YouTube changes frequently. Use startup script option 2, or `pip install --upgrade "yt-dlp[default]"`.
- **YouTube authentication:** browser cookies are the only viable method (OAuth deprecated). Use a throwaway account to avoid restrictions.
- **Rate limiting:** guest sessions ~300 videos/hour, authenticated ~2000 videos/hour. Use `--sleep-requests 5` to stay under limits.
- **Client selection:** `mweb` client is currently most reliable for YouTube (`--extractor-args "youtube:player_client=mweb"`).
- **Parallel processing:** long videos stream through worker pool instead of loading into RAM (prevents OOM).
- **Quality filters:** blur detection and quality scoring run automatically; frames failing either threshold are skipped, not saved. Only the `_watermarked` suffix is ever added (when --detect-watermarks is used). Thresholds default to 30/50 in **both** the CLI and the GUI — keep them in step.
- **Frame pipeline order matters:** crop black bars -> score -> threshold -> filter -> save. Scoring before the crop lets letterbox bars drag the score down and makes the filename describe a frame that was never written; scoring after the filters stamps a denoised frame with a worse blur number than it earned.
- **Output verbosity:** per-frame lines are `--verbose` only. A terminal gets a tqdm bar; piped output (the GUI streams this script's stdout into a Tk widget, where a carriage-return bar is unreadable) gets a periodic one-line summary instead. `_StatusReporter` owns that decision.
- **Dependency floors are the security surface:** pip will not upgrade an already-installed package that still satisfies a `>=` floor, so a stale floor is what long-lived installs keep running. CI audits both the resolved versions and the floors themselves; raise the floor when bumping, don't just rely on `>=`.
- **`keyframes` bypasses the frame pipeline** — it shells straight to FFmpeg, so quality/blur thresholds, watermark detection, post-processing filters and `--resume` do not apply to it (`--png` is honored).

## Security

This file is **public-safe by default**. Never add local paths, credentials, API keys, personal data, infrastructure details, or subscription info.

Before pushing: re-read this file and confirm it contains no local paths, credentials, or personal data.

**Browser cookies** contain authentication tokens. Cookies are read from browser profiles but not stored permanently by this tool. Use caution on shared systems.

## Maintenance

**Update rule:** When you change the architecture, build/test commands, or conventions, update this AGENTS.md in the same commit. Keep under 200 lines.

**CLAUDE.md:** One-line shim: `@AGENTS.md`.

**New-repo rule:** Create AGENTS.md in the first session a new repo is worked on.

**No-overlap rule:** Explanatory prose lives in one file. AGENTS.md = agent-facing summary; README.md = human/usage. Identical commands may be restated verbatim. Explanatory prose must not be duplicated — link instead.
