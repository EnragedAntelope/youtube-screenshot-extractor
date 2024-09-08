import yt_dlp
import cv2
import numpy as np
import os
import argparse
from datetime import datetime
import re
from PIL import Image

def sanitize_filename(filename):
    return re.sub(r'[^\w\-_.]', '_', filename)

def download_video(url, output_path):
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        video_title = info['title']
        ydl.download([url])
    return video_title

def calculate_quality_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sharpness
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    sharpness_norm = min(sharpness / 1000, 1.0)
    
    # Contrast
    contrast = np.std(gray)
    contrast_norm = contrast / 128
    
    # Brightness
    brightness = np.mean(gray)
    brightness_norm = brightness / 255
    
    # Calculate weighted score
    quality_score = (sharpness_norm * 0.5 + contrast_norm * 0.3 + brightness_norm * 0.2) * 100
    
    return quality_score

def is_black(pixel, threshold=10):
    return all(value < threshold for value in pixel[:3])

def remove_black_bars(image):
    width, height = image.size
    pixels = image.load()

    # Find top
    top = 0
    while top < height and all(is_black(pixels[x, top]) for x in range(width)):
        top += 1

    # Find bottom
    bottom = height - 1
    while bottom > top and all(is_black(pixels[x, bottom]) for x in range(width)):
        bottom -= 1

    # Find left
    left = 0
    while left < width and all(is_black(pixels[left, y]) for y in range(top, bottom + 1)):
        left += 1

    # Find right
    right = width - 1
    while right > left and all(is_black(pixels[right, y]) for y in range(top, bottom + 1)):
        right -= 1

    # Crop the image
    return image.crop((left, top, right + 1, bottom + 1))

def extract_frames(video_path, output_folder, interval_seconds=5, quality_threshold=50):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval_seconds)
    
    success, frame = video.read()
    count = 0
    frame_number = 0
    while success:
        if frame_number % frame_interval == 0:
            quality_score = calculate_quality_score(frame)
            print(f"Frame {frame_number}: Quality Score = {quality_score:.2f}")
            
            if quality_score > quality_threshold:
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                
                # Remove black bars
                pil_image = remove_black_bars(pil_image)
                
                frame_filename = os.path.join(output_folder, f"frame_{count:06d}.jpg")
                pil_image.save(frame_filename)
                print(f"Saved frame {frame_filename}")
                count += 1
            else:
                print(f"Skipped frame {frame_number} due to low quality")
        
        success, frame = video.read()
        frame_number += 1
    
    video.release()
    print(f"Extracted {count} high-quality frames.")

def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube video and extract high-quality frames with black bars removed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--interval", 
        type=float, 
        default=5.0, 
        help="Interval between frames in seconds. Lower values will extract more frames."
    )
    parser.add_argument(
        "--quality", 
        type=float, 
        default=12.0, 
        help="Quality threshold for frame selection (0-100). Higher values are more strict. 0 accepts all frames, 100 is very strict."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=None, 
        help="Custom output folder name. If not specified, a name will be generated based on the video title."
    )

    args = parser.parse_args()

    if args.quality < 0 or args.quality > 100:
        parser.error("Quality threshold must be between 0 and 100.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_path = f"downloaded_video_{timestamp}.mp4"
    
    video_title = download_video(args.url, video_path)
    sanitized_title = sanitize_filename(video_title)
    
    if args.output:
        output_folder = sanitize_filename(args.output)
    else:
        output_folder = f"screenshots_{sanitized_title}_{timestamp}"
    
    print(f"Video downloaded: {video_path}")
    print(f"Extracting frames to: {output_folder}")
    print(f"Quality threshold set to {args.quality:.1f} (Range: 0-100, Higher is stricter)")
    
    extract_frames(video_path, output_folder, args.interval, args.quality)
    print("Frame extraction complete.")

if __name__ == "__main__":
    main()
