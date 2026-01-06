# YouTube Screenshot Extractor and Dataset Gatherer

Extract high-quality frames from YouTube videos, local video files, or any yt-dlp supported source (1000+ sites). Useful for ML dataset preparation (LoRAs, checkpoints) or grabbing specific frames.
<img width="1887" height="1512" alt="image" src="https://github.com/user-attachments/assets/b604e3c9-152b-4a44-927a-a6e8767a5279" /><img width="1728" height="1539" alt="image" src="https://github.com/user-attachments/assets/64a44030-ccce-4034-9b35-e951a15ff5d0" />



## Features

- Download from YouTube and 1000+ sites via yt-dlp
- Process local video files
- Multiple extraction methods: interval, all frames, keyframes, scene detection
- Quality and blur filtering
- Automatic black bar removal
- Basic watermark detection
- Parallel processing and optional GPU acceleration
- Resume interrupted extractions
- Post-processing filters (gradfun, deblock, deband)
- **GUI and command-line interfaces**

## Quick Start (Windows)

**Double-click `START.bat`** and select:
1. **Initial Setup** - First time only, creates environment and installs dependencies
2. **Install Deno** - **Required for YouTube** (other sites work without it)
3. **Launch GUI** - Start the graphical interface

The menu also offers yt-dlp updates and optional GPU support.

## Requirements

- Python 3.10+
- [Deno](https://deno.land/) - **Required for YouTube downloads** (use `START.bat` option 3, or `winget install DenoLand.Deno`)
- [FFmpeg](https://ffmpeg.org/download.html) - Required for keyframe extraction and filters
- PyCUDA (optional) - for NVIDIA GPU acceleration

**Important:** Keep yt-dlp updated regularly. Use `START.bat` option 2, or run: `pip install --upgrade yt-dlp`

## Manual Installation

If you prefer not to use `START.bat`:

```bash
git clone https://github.com/EnragedAntelope/youtube-screenshot-extractor.git
cd youtube-screenshot-extractor
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt

# Install Deno (required for YouTube)
winget install DenoLand.Deno
```

## Usage

### GUI (Recommended)

```bash
python youtube-screenshot-gui.py
```

The GUI provides all options with helpful tooltips, good defaults pre-selected, and a clean interface.

### Command Line

```bash
# Extract frames using scene detection from YouTube
python youtube-screenshot-script.py https://www.youtube.com/watch?v=VIDEO_ID --method scene

# Extract keyframes from local file
python youtube-screenshot-script.py video.mp4 --method keyframes

# Optimal quality extraction with filters
python youtube-screenshot-script.py video.mp4 --quality 50 --blur 100 --detect-watermarks --deblock --thumbnail
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--method` | `interval`, `all`, `keyframes`, or `scene` | interval |
| `--interval` | Seconds between frames (interval method only) | 5.0 |
| `--quality` | Quality threshold 0-100 (higher = stricter) | 12.0 |
| `--blur` | Blur threshold (higher = less blur allowed) | 10.0 |
| `--max-resolution` | Limit download resolution (e.g., 720, 1080) | best |
| `--output` | Custom output folder name | auto |
| `--png` | Save as PNG instead of JPG | JPG |
| `--detect-watermarks` | Enable watermark detection | off |
| `--watermark-threshold` | Watermark sensitivity 0-1 | 0.8 |
| `--use-gpu` | Enable GPU acceleration | off |
| `--fast-scene` | Faster but less accurate scene detection | off |
| `--resume` | Resume interrupted extraction | off |
| `--thumbnail` | Generate 3x3 thumbnail montage | off |
| `--verbose` | Detailed logging | off |
| `--dry-run` | Preview without processing | off |
| `--config` | Load settings from JSON file | none |
| `--gradfun` | Reduce color banding (subtle) | off |
| `--deblock` | Reduce compression artifacts | off |
| `--deband` | Reduce color banding (aggressive) | off |

## Output

Frames are saved as: `frame_NNNNNN_qXX_bYY[_watermarked].(jpg|png)`
- `NNNNNN`: Frame number
- `XX`: Quality score (0-99, higher is better)
- `YY`: Blur score (higher = sharper)
- `_watermarked`: Added if watermark detected

## Tips

- **Speed**: `keyframes` is fastest. `scene` finds natural cuts. `interval` and `all` can be very slow.
- **Quality tuning**: Start with `--quality 30 --blur 50` and adjust based on results.
- **Large videos**: Use `--resume` and `--max-resolution 1080` to manage long processes and avoid rate limiting.
- **Filters**: Use `--gradfun` for subtle banding, `--deband` for severe banding. Filters increase processing time.
- **Other sites**: Most of the 1000+ sites yt-dlp supports will work. Some may not support all resolution options.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| YouTube download fails | Install Deno: `winget install DenoLand.Deno` (required since Nov 2025) |
| Authentication/download errors | Update yt-dlp: `pip install --upgrade yt-dlp` |
| "Format not available" error | Remove resolution limit or try a different source - some sites have limited formats |
| No frames extracted | Lower thresholds: `--quality 20 --blur 30` |
| Keyframe extraction fails | Ensure FFmpeg is installed and in PATH |
| Scene detection slow/crashes | Use `--fast-scene` or process shorter segments |
| GPU not working | Verify CUDA and PyCUDA installation, or remove `--use-gpu` |
| False watermark positives | Increase threshold: `--watermark-threshold 0.9` |
| Process dies on large videos | Use `--resume`, check disk space |

## GPU Acceleration (Optional)

For NVIDIA GPUs, install PyCUDA for faster processing:

1. Install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)
2. Run `START.bat` option 4, or: `pip install pycuda`

**Note:** PyCUDA installation takes 5-10 minutes. Only recommended if processing many videos. Modern CPUs are usually fast enough.

## License

MIT License - see [LICENSE](LICENSE)
