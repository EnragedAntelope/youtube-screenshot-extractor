# YouTube Screenshot Extractor

## Overview

This Python script is a powerful tool for extracting high-quality screenshots from YouTube videos or local video files. It's particularly useful for preparing datasets for machine learning projects, such as training Loras or checkpoints.

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
- Customizable output options

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

Note: The scenedetect library is now included in the requirements.txt file. If you encounter any issues with scene detection, ensure you have the latest version installed:
```
pip install --upgrade scenedetect
```

## Usage

Basic syntax:
```
python youtube-screenshot-script.py [SOURCE] [OPTIONS]
```

Where `[SOURCE]` can be either a YouTube URL or a path to a local video file.

To see all available options and their descriptions, use the --help flag:
```
python youtube-screenshot-script.py --help
```

### Options

- `--method {interval,all,keyframes,scene}`: Frame extraction method (default: interval)
  - `interval`: Extract frames at specified intervals
  - `all`: Extract all frames
  - `keyframes`: Extract only keyframes
  - `scene`: Use scene detection to extract frames
- `--interval INTERVAL`: Interval between frames in seconds (default: 5.0, only used with 'interval' method)
- `--quality QUALITY`: Quality threshold for frame selection (0-100, default: 50.0)
- `--blur BLUR`: Blur threshold for frame selection (default: 100.0)
- `--output OUTPUT`: Custom output folder name

### Examples

1. Extract frames every 5 seconds from a YouTube video:
   ```
   python youtube-screenshot-script.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
   ```

2. Extract frames every 10 seconds with a lower quality threshold:
   ```
   python youtube-screenshot-script.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --interval 10 --quality 30
   ```

3. Extract keyframes from a local video file:
   ```
   python youtube-screenshot-script.py path/to/your/video.mp4 --method keyframes
   ```

4. Use scene detection on a YouTube video:
   ```
   python youtube-screenshot-script.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --method scene
   ```

## What to Expect

- **Processing Time**: 
  - The 'interval' and 'all' methods are generally the fastest.
  - Keyframe extraction is moderately fast but requires FFmpeg.
  - Scene detection is the most time-consuming method but can provide excellent results for videos with distinct scene changes.

- **Quality Threshold**: 
  - The default quality threshold (50.0) may be too high for some YouTube videos. 
  - For lower quality videos, try reducing the quality threshold (e.g., --quality 30 or lower).
  - Experiment with different values to find the best balance between quantity and quality of extracted frames.

- **Output**: 
  - Frames are saved as JPEG images in a folder named after the video title (or custom name if specified).
  - Each frame filename includes its sequence number.

- **Blur Detection**: 
  - The script includes blur detection to filter out blurry frames.
  - Adjust the blur threshold if you're getting too few or too many frames.

- **Black Bar Removal**: 
  - The script automatically attempts to remove black bars from the extracted frames.
  - This feature works best on videos with consistent black bars.

## Tips

- For quick results, start with the 'interval' method and adjust the interval and quality threshold as needed.
- Use the 'scene' method for videos with distinct scene changes, but be prepared for longer processing times.
- When working with lower quality YouTube videos, start with a lower quality threshold (e.g., --quality 20) and adjust as needed.
- If you're extracting frames for machine learning datasets, consider using a combination of methods to get a diverse set of high-quality frames.

## Troubleshooting

- If you encounter issues with keyframe extraction, ensure FFmpeg is properly installed and accessible in your system PATH.
- For scene detection issues, make sure the scenedetect library is installed (`pip install scenedetect`).
- If you're not getting any frames, try lowering the quality and blur thresholds.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
