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
- `--quality QUALITY`: Quality threshold for frame selection (0-100, default: 30.0). Frames with quality scores above this threshold will be kept.
- `--blur BLUR`: Blur threshold for frame selection (default: 100.0). Frames with blur scores above this threshold will be kept.
- `--output OUTPUT`: Custom output folder name

### Examples

1. Extract frames every 5 seconds from a YouTube video:
   ```
   python youtube-screenshot-script.py https://www.youtube.com/watch?v=dQw4w9WgXcQ
   ```

2. Extract frames every 10 seconds with a higher quality threshold:
   ```
   python youtube-screenshot-script.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --interval 10 --quality 40
   ```

3. Extract frames with a lower quality threshold for low-quality videos:
   ```
   python youtube-screenshot-script.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --quality 12
   ```

4. Extract keyframes from a local video file:
   ```
   python youtube-screenshot-script.py path/to/your/video.mp4 --method keyframes
   ```

5. Use scene detection on a YouTube video:
   ```
   python youtube-screenshot-script.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --method scene
   ```

## What to Expect

- **Processing Time**: 
  - The 'interval' and 'all' methods are generally the fastest.
  - Keyframe extraction is moderately fast but requires FFmpeg.
  - Scene detection is the most time-consuming method but can provide excellent results for videos with distinct scene changes.

- **Quality Threshold**: 
  - The default quality threshold (30.0) is set to work well with most YouTube videos.
  - For lower quality videos, you may need to reduce the quality threshold further (e.g., --quality 20 or even down to the 10-12 range).
  - Frames with quality scores above the set threshold will be kept.
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

- Start with the default quality threshold (30.0) and adjust as needed. For high-quality videos, you might increase it slightly, while for lower quality videos, you may need to decrease it significantly.
- Use the 'interval' method for quick results, and adjust the interval and quality threshold to get the desired number of frames.
- The 'scene' method is excellent for videos with distinct scene changes but may take longer to process.
- When working with lower quality YouTube videos, start with a lower quality threshold (e.g., --quality 20 or even 12) and adjust as needed.
- If you're extracting frames for machine learning datasets, consider using a combination of methods to get a diverse set of high-quality frames.

## Troubleshooting

- If you encounter issues with keyframe extraction, ensure FFmpeg is properly installed and accessible in your system PATH.
- For scene detection issues, make sure the scenedetect library is installed (`pip install scenedetect`).
- If you're not getting any frames, try lowering the quality threshold. Some videos may require values as low as 10-12.
- If the script is extracting too many similar frames, try increasing the interval or quality threshold.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
