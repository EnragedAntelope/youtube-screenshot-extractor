# YouTube Screenshot Extractor and Dataset Gatherer

Extract high-quality frames from YouTube videos, local video files, or any yt-dlp supported source (1000+ sites). Useful for ML dataset preparation (LoRAs, checkpoints) or grabbing specific frames.
<img width="1909" height="1624" alt="image" src="https://github.com/user-attachments/assets/4a98db38-902b-4143-b184-63a6c833fb4a" />
<img width="1728" height="1539" alt="image" src="https://github.com/user-attachments/assets/64a44030-ccce-4034-9b35-e951a15ff5d0" />



## Features

- Download from YouTube and 1000+ sites via yt-dlp
- Process local video files
- Multiple extraction methods: interval, all frames, keyframes, scene detection
- Quality and blur filtering
- Automatic black bar removal
- Basic watermark detection
- Parallel processing with bounded memory use (handles long videos)
- Resume interrupted extractions
- Post-processing filters (gradfun, deblock, deband)
- **GUI and command-line interfaces**
- **YouTube authentication support** (cookies for age-restricted/private videos)
- **Rate limiting protection** (avoid IP bans when processing multiple videos)

## Quick Start

### Windows
**Double-click `START.bat`** and use the menu:
- **[1] Initial Setup** - First time only, creates environment and installs dependencies
- **[3] Install Deno** - **Required for YouTube** (other sites work without it)
- **[4] Install FFmpeg** - **Required for keyframes and some filters**
- **[5] Launch GUI** - Start the graphical interface

The menu also offers yt-dlp updates (option 2) and command-line help (option 6).

### macOS / Linux
**Run `./start.sh`** in terminal and use the menu:
- **[1] Initial Setup** - First time only, creates environment and installs dependencies
- **[3] Install Deno** - **Required for YouTube** (other sites work without it)
- **[4] Install FFmpeg** - **Required for keyframes and some filters**
- **[5] Launch GUI** - Start the graphical interface

The menu also offers yt-dlp updates (option 2) and command-line help (option 6).

## Requirements

- Python 3.10+
- [Deno](https://deno.land/) - **Required for YouTube downloads**
  - Windows: Use `START.bat` option 3, or `winget install DenoLand.Deno`
  - macOS: Use `./start.sh` option 3, or `brew install deno`
  - Linux: Use `./start.sh` option 3, or `curl -fsSL https://deno.land/install.sh | sh`
- [FFmpeg](https://ffmpeg.org/download.html) - **Required for keyframes, gradfun/deband filters, and audio merging**
  - Windows: Use `START.bat` option 4, or `winget install Gyan.FFmpeg`
  - macOS: Use `./start.sh` option 4, or `brew install ffmpeg`
  - Linux: Use `./start.sh` option 4, or `sudo apt install ffmpeg` (Ubuntu/Debian)
  - Note: Scene detection and deblock filter work without FFmpeg

**Important:** Keep yt-dlp updated regularly — YouTube compatibility breaks frequently. Use startup script option 2, or run: `pip install --upgrade "yt-dlp[default]"`

## Manual Installation

If you prefer not to use the startup scripts:

### Windows
```bash
git clone https://github.com/EnragedAntelope/youtube-screenshot-extractor.git
cd youtube-screenshot-extractor
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

# Keep yt-dlp current (run periodically for YouTube compatibility)
pip install --upgrade "yt-dlp[default]"

# Install Deno (required for YouTube)
winget install DenoLand.Deno

# Install FFmpeg (required for keyframes/filters)
winget install Gyan.FFmpeg
```

### macOS / Linux
```bash
git clone https://github.com/EnragedAntelope/youtube-screenshot-extractor.git
cd youtube-screenshot-extractor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Keep yt-dlp current (run periodically for YouTube compatibility)
pip install --upgrade "yt-dlp[default]"

# Install Deno (required for YouTube)
# macOS:
brew install deno
# Linux:
curl -fsSL https://deno.land/install.sh | sh

# Install FFmpeg (required for keyframes/filters)
# macOS:
brew install ffmpeg
# Linux (Ubuntu/Debian):
sudo apt install ffmpeg
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

# Extract from age-restricted video (requires authentication)
python youtube-screenshot-script.py "YOUTUBE_URL" --cookies-from-browser firefox

# Process multiple videos with rate limiting to avoid bans
python youtube-screenshot-script.py "URL" --sleep-requests 5
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--method` | `interval`, `all`, `keyframes`, or `scene` | interval |
| `--interval` | Seconds between frames (interval method only) | 5.0 |
| `--quality` | Quality threshold 0-100 (higher = stricter) | 12.0 |
| `--blur` | Blur threshold (higher = less blur allowed) | 10.0 |
| `--max-resolution` | Limit download quality (e.g., 720, 1080) | best |
| `--output` | Custom output folder name | auto |
| `--png` | Save as PNG instead of JPG | JPG |
| `--detect-watermarks` | Enable watermark detection | off |
| `--watermark-threshold` | Watermark sensitivity 0-1 | 0.8 |
| `--fast-scene` | Faster but less accurate scene detection | off |
| `--resume` | Resume interrupted extraction | off |
| `--thumbnail` | Generate 3x3 thumbnail montage | off |
| `--verbose` | Detailed logging | off |
| `--dry-run` | Preview without downloading or processing | off |
| `--disable-parallel` | Process frames one at a time | off |
| `--config` | Load settings from JSON file | none |
| `--gradfun` | Reduce color banding (subtle) | off |
| `--deblock` | Reduce compression artifacts | off |
| `--deband` | Reduce color banding (aggressive) | off |
| `--cookies-from-browser` | Use cookies from browser (firefox, chrome, etc.) | none |
| `--cookies` | Path to cookies file (Netscape format) | none |
| `--sleep-requests` | Delay in seconds between requests | 0 |
| `--extractor-args` | Additional yt-dlp extractor arguments | none |

## Output

Frames are saved as: `frame_NNNNNN_qXX_bYY[_watermarked].(jpg|png)`
- `NNNNNN`: Frame number
- `XX`: Quality score (0-99, higher is better)
- `YY`: Blur score (higher = sharper)
- `_watermarked`: Added if watermark detected

## Tips

- **Speed**: `keyframes` is fastest. `scene` finds natural cuts. `interval` and `all` can be very slow.
- **Quality tuning**: Start with `--quality 30 --blur 50` and adjust based on results.
- **Large videos**: Use `--resume` and `--max-resolution 1080` to manage long processes.
- **YouTube authentication**: Always use `--cookies-from-browser firefox` for best results with YouTube.
- **Avoiding rate limits**: Use `--sleep-requests 5` when processing multiple videos in a batch.
- **Filters**: Use `--gradfun` for subtle banding, `--deband` for severe banding. Filters increase processing time.
- **Other sites**: Most of the 1000+ sites yt-dlp supports will work. Some may not support all resolution options.

## YouTube Authentication Guide

### Using Browser Cookies (Recommended)

For the best experience with YouTube videos, use browser cookies:

1. **Login to YouTube** in your browser (Firefox recommended on Windows)
2. **Run the tool with authentication**:
   ```bash
   python youtube-screenshot-script.py "YOUTUBE_URL" --cookies-from-browser firefox
   ```

3. **For age-restricted videos**, you must be logged into a YouTube account in that browser

### Rate Limiting Best Practices

YouTube enforces rate limits to prevent abuse:
- **Guest users**: ~300 videos/hour
- **Authenticated users**: ~2000 videos/hour

To avoid hitting these limits:

1. **Use rate limiting**: `--sleep-requests 5` adds a 5-second delay between requests
2. **Download at lower resolutions**: `--max-resolution 720` downloads faster
3. **Process in batches**: Don't queue hundreds of videos at once

### Security Considerations

⚠️ **Warning**: Using your YouTube account with yt-dlp carries a small risk of account restrictions. To minimize risk:

- Use a throwaway/secondary YouTube account for downloading
- Don't download excessive amounts (stay under rate limits)
- Keep yt-dlp updated to ensure you're using the latest, safest methods

## Troubleshooting

| Problem | Solution |
|---------|----------|
| YouTube download fails | 1. Update yt-dlp: `pip install --upgrade "yt-dlp[default]"`<br>2. Use `--cookies-from-browser firefox` for PO Token issues<br>3. Install/update Deno (required):<br>Windows: `winget install DenoLand.Deno`<br>macOS: `brew install deno`<br>Linux: `curl -fsSL https://deno.land/install.sh \| sh` |
| HTTP 403 / "Forbidden" error | YouTube requires authentication. Use `--cookies-from-browser firefox` or `--cookies-from-browser chrome` |
| Age-restricted video fails | Use `--cookies-from-browser firefox` (must be logged into YouTube in that browser) |
| Rate limiting / "This content isn't available" | Add delay between requests: `--sleep-requests 5` and use `--max-resolution 720` |
| "Format not available" error | Remove resolution limit or try a different source - some sites have limited formats |
| No frames extracted | Lower thresholds: `--quality 20 --blur 30` |
| Keyframe extraction fails | Ensure FFmpeg is installed and in PATH |
| Scene detection slow/crashes | Use `--fast-scene` or process shorter segments |
| False watermark positives | Increase threshold: `--watermark-threshold 0.9` |
| Process dies on large videos | Use `--resume`, check disk space |

### Understanding YouTube PO Tokens

YouTube has implemented **PO (Proof of Origin) Tokens** as an anti-bot measure. Without proper authentication, you may encounter HTTP 403 errors when downloading videos.

**What this means for you:**
- Many YouTube videos now require browser cookies to download
- Age-restricted and private videos always require authentication
- Rate limits are enforced: ~300 videos/hour for guests, ~2000 for authenticated users

**Solutions:**

1. **For most videos**: Use `--cookies-from-browser firefox` (or `chrome`, `edge`)
   - You must be logged into YouTube in that browser
   - Firefox is recommended on Windows (Chrome encrypts cookies)

2. **For age-restricted/private videos**: 
   - Use `--cookies-from-browser firefox` with a browser where you're logged into YouTube
   - Consider using a throwaway account to avoid risking your main account

3. **To avoid rate limiting**:
   - Use `--sleep-requests 5` to add a 5-second delay between requests
   - Use `--max-resolution 720` to download smaller files (faster, less likely to trigger limits)
   - Process videos in smaller batches

4. **Keep yt-dlp updated**:
   ```bash
   pip install --upgrade "yt-dlp[default]"
   ```
   YouTube changes their systems frequently, and yt-dlp updates regularly to keep up.

## Note on GPU Acceleration

Earlier versions offered a `--use-gpu` flag (PyCUDA). It has been removed: the GPU path did not actually accelerate anything and altered image brightness as a side effect. All processing is CPU-based and parallelized; the flag is still accepted for backwards compatibility but does nothing.

## License

MIT License - see [LICENSE](LICENSE)
