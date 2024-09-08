# YouTube Screenshot Extractor

This script downloads a YouTube video and extracts high-quality frames at specified intervals, removing black bars from the screenshots.

## Features

- Download YouTube videos using yt-dlp
- Extract frames at specified intervals
- Calculate quality scores for each frame
- Remove black bars from extracted frames
- Save high-quality frames as JPEG images

## Requirements

- Python 3.6+
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

## Usage

Run the script with the following command:

```
python youtube_screenshot_script.py [YouTube URL] [options]
```

### Options

- `--interval INTERVAL`: Interval between frames in seconds (default: 5.0)
- `--quality QUALITY`: Quality threshold for frame selection (default: 10.0)
- `--output OUTPUT`: Custom output folder name

### Example

```
python youtube_screenshot_script.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --interval 10 --quality 15 --output my_screenshots
```

This command will:
1. Download the video from the given URL
2. Extract frames every 10 seconds
3. Save frames with a quality score above 15
4. Store the extracted frames in a folder named "my_screenshots"

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
