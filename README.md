# YouTube Screenshot Extractor (But Not Just for YouTube!) and Dataset Gatherer

## Overview

This Python script is a versatile tool for extracting high-quality screenshots from YouTube videos, local video files, or any other video source you can think of! It's particularly useful for preparing datasets for machine learning projects, such as training Loras or checkpoints, or just for grabbing that perfect frame from your favorite movie.

## Features

- Download YouTube videos using yt-dlp
- Process local video files
- Multiple frame extraction methods:
  - Interval-based extraction
  - All frames extraction
  - Keyframe extraction
  - Scene detection
- Quality assessment of frames
- Blur detection
- Automatic removal of black bars
- Basic watermark detection
- Parallel processing for faster execution
- GPU acceleration (if available)
- Resume interrupted extractions
- Generate thumbnail montages
- Customizable output options (JPG or PNG)
- Detailed logging and dry-run option
- Load settings from a configuration file

## Requirements

- Python 3.6+
- FFmpeg (required for keyframe extraction)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/EnragedAntelope/youtube-screenshot-extractor.git
   cd youtube-screenshot-extractor
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install FFmpeg (required for keyframe extraction):
   - Follow the instructions at: https://ffmpeg.org/download.html

## Usage

Basic syntax:
```
python youtube-screenshot-script.py [SOURCE] [OPTIONS]
```

Where `[SOURCE]` can be either a YouTube URL or a path to a local video file.

### Quick Start Example

Extract frames every 5 seconds from a YouTube video and save them to a custom folder:
```
python youtube-screenshot-script.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --output my_awesome_screenshots
```

It's that easy! Now let's dive into all the configurable options.

### Options

- `--method {interval,all,keyframes,scene}`: Frame extraction method (default: interval)
- `--interval INTERVAL`: Interval between frames in seconds (default: 5.0, only used with 'interval' method)
- `--quality QUALITY`: Quality threshold for frame selection (0-100, default: 12.0)
- `--blur BLUR`: Blur threshold for frame selection (default: 10.0)
- `--detect-watermarks`: Enable basic watermark detection
- `--watermark-threshold THRESHOLD`: Watermark detection sensitivity (0-1, default: 0.8)
- `--max-resolution RESOLUTION`: Maximum resolution for YouTube video download (e.g., 720, 1080). Ignored for local files.
- `--output OUTPUT`: Custom output folder name
- `--png`: Save frames as PNG instead of JPG
- `--disable-parallel`: Disable parallel processing of frames
- `--use-gpu`: Use GPU acceleration if available
- `--fast-scene`: Use fast mode for scene detection (less accurate results)
- `--resume`: Resume an interrupted extraction process
- `--thumbnail`: Generate a thumbnail montage of extracted frames
- `--verbose`: Enable detailed logging
- `--dry-run`: Show what would be done without actually processing
- `--config CONFIG`: Load settings from a configuration file

### Examples

1. Extract keyframes from a local video file:
   ```
   python youtube-screenshot-script.py path/to/your/video.mp4 --method keyframes
   ```

2. Use scene detection on a YouTube video with custom output folder:
   ```
   python youtube-screenshot-script.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --method scene --output my_scene_shots
   ```

3. Extract frames every 10 seconds with a lower quality threshold and PNG output:
   ```
   python youtube-screenshot-script.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --interval 10 --quality 30 --png
   ```

4. Use GPU acceleration and generate a thumbnail montage:
   ```
   python youtube-screenshot-script.py path/to/your/video.mp4 --use-gpu --thumbnail
   ```
5. Download a YouTube video at a maximum resolution of 720p and extract frames:
   ```
   python youtube-screenshot-script.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --max-resolution 720
   ```

## What to Expect

- **Processing Time**: 
  - The 'interval' and 'keyframes' methods are generally the fastest.
  - The 'all' frames method can be time-consuming for longer videos.
  - Scene detection is typically the most time-consuming method, especially for longer videos.
  - GPU acceleration can significantly speed up processing for all methods.
	
- **Video Resolution**: 
  - When downloading YouTube videos, the script will use the highest available quality by default. Use the `--max-resolution` option to limit the download quality if needed.
  - For local video files, the original resolution is always used, and the `--max-resolution` option is ignored.

- **Output**: Frames are saved as JPG or PNG images in the specified output folder (or a default folder named after the video title).

- **Frame Filenames**: Follow the format `frame_NNNNNN_qXX_bYY[_watermarked].(jpg|png)`, where:
  - `NNNNNN`: Frame number
  - `XX`: Quality score (0-99, higher is better)
  - `YY`: Blur score (higher numbers indicate less blur)
  - `_watermarked`: Suffix added if a watermark is detected

## Tips

- Start with the 'interval' method for quick results. Use scene detection for videos with distinct scene changes, but be prepared for longer processing times.
- Use the `--dry-run` option to preview the extraction process without actually saving frames.
- For large videos or long-running processes:
  - Use the `--resume` option in case the process is interrupted.
  - Consider using the `--max-resolution` option to limit download quality for faster processing and to conserve storage space.
    - Note: The --max-resolution option only applies when downloading YouTube videos. It has no effect when processing local video files.
- To optimize quality and performance:
  - Experiment with different quality and blur thresholds to find the best balance for your needs. Start with lower thresholds (e.g., `--quality 30 --blur 50`) and adjust as needed.
  - If processing is too slow, try disabling watermark detection or using GPU acceleration if available.
  - Use the `--fast-scene` option for quicker (but less accurate) scene detection results.

## Troubleshooting

- **No frames extracted**: 
  - Lower the quality and blur thresholds (e.g., `--quality 20 --blur 30`).
  - Check if the video file is corrupted or if yt-dlp failed to download it properly.

- **Keyframe extraction issues**: 
  - Ensure FFmpeg is properly installed and accessible in your system PATH.
  - Try updating FFmpeg to the latest version.

- **Scene detection extremely slow or crashing**:
  - Use the `--fast-scene` option for quicker but less accurate results.
  - Try processing a shorter segment of the video first to test.
  - Ensure you have enough RAM available.

- **GPU acceleration not working**:
  - Verify that you have CUDA-compatible GPU and the necessary CUDA libraries installed.
  - Check if PyCUDA is installed correctly (`pip install pycuda`).
  - If issues persist, fall back to CPU processing by removing the `--use-gpu` flag.

- **Watermark detection producing false positives**:
  - Adjust the watermark threshold (e.g., `--watermark-threshold 0.9` for stricter detection).
  - If unnecessary, disable watermark detection by removing the `--detect-watermarks` flag.

- **Process dies unexpectedly for large videos**:
  - Use the `--resume` option to continue from where it left off.
  - Try processing the video in smaller segments.
  - Ensure you have enough free disk space.

- **Low quality or blurry output**:
  - Increase the quality and blur thresholds.
  - Check if the source video is of sufficient quality.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
