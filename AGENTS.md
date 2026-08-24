# AGENTS.md — youtube-screenshot-extractor

Pull clean, high-quality still frames from videos — YouTube, 1000+ other sites via yt-dlp, or local files. Downloads (if needed), finds best frames, filters blurry/low-quality ones, crops black bars, saves the rest. Built for gathering training images for LoRAs/checkpoints. GUI + CLI, Python 3.10+, requires Deno (YouTube) and FFmpeg (keyframes).

**Deep reference: Previous CLAUDE.md contained implementation notes about YouTube PO Tokens, rate limiting, and authentication — that content is now archived in the project history**

## Current state

_Last verified: 2026-08-24_

- **Status:** working and maintained, no version number and no release tags — `git log` is the only version record. The recent history is a run of audits fixing silently-broken options rather than new features.
- **Works:** all four extraction methods (interval, every frame, keyframes, scene detection); automatic blur/quality filtering, black-bar cropping and watermark flagging; YouTube authentication via browser cookies plus request-rate throttling; resume for large extractions; parallel worker-pool processing that streams frames instead of loading the whole video; GUI and CLI at parity.
- **In progress:** nothing — recent history is three audit rounds: dependency/security floors plus the first test suite and CI; a robustness/parity pass (Stop kills the whole process tree, failed downloads clean up their partial files, `--png` honored by keyframes, FFmpeg filter frames piped instead of temp-filed, GUI blur range matched to the CLI); then a correctness pass (scene-frame numbering, GUI keyframes gating, typed-entry validation, CI job timeouts and xvfb GUI coverage).
- **Known gaps / next steps:** tests cover the pure helpers, `process_frame`, and the GUI's behaviour against a real Tk tree (`tests/`, run with `pytest`; CI runs them under xvfb) — **downloading is still verified only by hand**, since nothing in CI touches the network; **YouTube extraction is inherently fragile** — yt-dlp must be kept current (launcher option 2, or `pip install --upgrade "yt-dlp[default]"`), and the working client selection changes over time; Deno is required for YouTube and FFmpeg for keyframes, so a partial install silently limits which methods work; the rate-limit and client-selection notes in *Conventions & gotchas* are the most perishable content in this file — re-verify them before trusting them.
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
| `tests/` | `test_extractor.py` (CLI helpers, static GUI/CLI parity scans), `test_gui.py` (real Tk widget tree) |
| `.github/workflows/ci.yml` | Lint, byte-compile, unit tests, CLI smoke run, dependency audit, shellcheck |
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
- **GUI progress contract:** the GUI sets `YSE_PROGRESS=1` for the CLI subprocess; `_StatusReporter` then emits one `@@PROGRESS done=N total=M saved=S skipped=K` line per processed frame, which the GUI parses, strips from the log, and renders as a determinate bar. Terminals never emit or see these lines.
- **Dependency floors are the security surface:** pip will not upgrade an already-installed package that still satisfies a `>=` floor, so a stale floor is what long-lived installs keep running. CI audits both the resolved versions and the floors themselves; raise the floor when bumping, don't just rely on `>=`.
- **`keyframes` bypasses the frame pipeline** — it shells straight to FFmpeg, so quality/blur thresholds, watermark detection, post-processing filters, `--resume` and parallelism do not apply to it (`--png` and `--thumbnail` are honored). The GUI disables those controls and omits their flags while keyframes is selected (`_sync_pipeline_controls`); keep the two in step when adding a pipeline option.
- **Frame counts must have no gaps.** `_ProgressTracker` only advances across a contiguous run of completed indices, so the count `frame_generator` yields numbers the frames it actually produced, never their position in the source. A skipped number strands every later frame in the pending map: totals under-report and `--resume` checkpoints at the gap.
- **GUI numeric entries are typeable.** Spinbox-backed `DoubleVar`/`IntVar` raise `TclError` on `get()` when the box holds text, and an escaped exception in a button callback has nowhere to print under pythonw — the button just looks dead. Read them through `_read_number`.
- **CI jobs carry `timeout-minutes`.** GitHub's default is six hours, and a hung `apt-get` step once burned all of it.

## Security

This file is **public-safe by default**. Never add local paths, credentials, API keys, personal data, infrastructure details, or subscription info.

Before pushing: re-read this file and confirm it contains no local paths, credentials, or personal data.

**Browser cookies** contain authentication tokens. Cookies are read from browser profiles but not stored permanently by this tool. Use caution on shared systems.

## Maintenance

**Update rule:** When you change the architecture, build/test commands, or conventions, update this AGENTS.md in the same commit. Keep under 200 lines.

**CLAUDE.md:** One-line shim: `@AGENTS.md`.

**New-repo rule:** Create AGENTS.md in the first session a new repo is worked on.

**No-overlap rule:** Explanatory prose lives in one file. AGENTS.md = agent-facing summary; README.md = human/usage. Identical commands may be restated verbatim. Explanatory prose must not be duplicated — link instead.
