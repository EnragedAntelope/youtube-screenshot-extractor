import yt_dlp
import cv2
import numpy as np
import os
import argparse
from datetime import datetime
import re
from PIL import Image
import subprocess
import sys

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("Error: FFmpeg is not installed or not in the system PATH.")
        print("Please install FFmpeg to use the keyframe extraction feature.")
        print("Installation instructions: https://ffmpeg.org/download.html")
        sys.exit(1)

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
    
    # Sharpness using Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    sharpness_norm = min(sharpness / 1000, 1.0)
    
    # Edge strength using Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
    edge_strength_norm = min(edge_strength / 100, 1.0)
    
    # Contrast
    contrast = np.std(gray)
    contrast_norm = contrast / 128
    
    # Brightness
    brightness = np.mean(gray)
    brightness_norm = brightness / 255
    
    # Calculate weighted score
    quality_score = (sharpness_norm * 0.4 + edge_strength_norm * 0.3 + contrast_norm * 0.2 + brightness_norm * 0.1) * 100
    
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

def extract_frames(video_path, output_folder, method='interval', interval_seconds=5, quality_threshold=30, blur_threshold=100):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    
    if method == 'interval':
        frame_interval = int(fps * interval_seconds)
    elif method == 'all':
        frame_interval = 1
    elif method == 'keyframes':
        check_ffmpeg()
        ffmpeg_command = (
            f"ffmpeg -i {video_path} "
            f"-vf select='eq(pict_type,PICT_TYPE_I)',"
            f"scale=in_range=full:out_range=tv,"
            f"zscale=t=linear:npl=100:m=bt709:r=tv,"
            f"format=yuv420p "
            f"-fps_mode vfr "
            f"-q:v 2 {output_folder}/keyframe_%03d.jpg"
        )
        os.system(ffmpeg_command)
        video.release()
        print("Keyframe extraction complete.")
        return
    elif method == 'scene':
        try:
            from scenedetect import detect, ContentDetector
        except ImportError:
            print("Error: The 'scene' method requires the scenedetect library.")
            print("Please install it manually using:")
            print("pip install scenedetect")
            sys.exit(1)
        
        scene_list = detect(video_path, ContentDetector())
        for i, scene in enumerate(scene_list):
            frame_number = scene[0].frame_num
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = video.read()
            if ret:
                process_frame(frame, output_folder, i, quality_threshold, blur_threshold)
        video.release()
        print("Scene detection and extraction complete.")
        return
    
    success, frame = video.read()
    count = 0
    frame_number = 0
    while success:
        if frame_number % frame_interval == 0:
            process_frame(frame, output_folder, count, quality_threshold, blur_threshold)
            count += 1
        
        success, frame = video.read()
        frame_number += 1
    
    video.release()
    print(f"Extracted {count} frames.")

def process_frame(frame, output_folder, count, quality_threshold, blur_threshold):
    quality_score = calculate_quality_score(frame)
    print(f"Frame {count}: Quality Score = {quality_score:.2f}")
    
    # Additional blur check
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    if quality_score >= quality_threshold and laplacian_var >= blur_threshold:
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Remove black bars
        pil_image = remove_black_bars(pil_image)
        
        frame_filename = os.path.join(output_folder, f"frame_{count:06d}.jpg")
        pil_image.save(frame_filename)
        print(f"Saved frame {frame_filename}")
    else:
        print(f"Skipped frame {count} due to low quality or blur")

def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube video or process local video file and extract high-quality frames with black bars removed.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("source", help="YouTube video URL or path to local video file")
    parser.add_argument(
        "--method", 
        choices=['interval', 'all', 'keyframes', 'scene'],
        default='interval',
        help="Frame extraction method: \
            'interval' for every n seconds, \
            'all' for all frames, \
            'keyframes' for only keyframes, \
            'scene' for scene detection"
    )
    parser.add_argument(
        "--interval", 
        type=float, 
        default=5.0, 
        help="Interval between frames in seconds (only used with 'interval' method). Lower values will extract more frames."
    )
    parser.add_argument(
        "--quality", 
        type=float, 
        default=30.0,
        help="Quality threshold for frame selection (0-100). Higher values are more strict. 0 accepts all frames, 100 is very strict."
    )
    parser.add_argument(
        "--blur", 
        type=float, 
        default=100.0,
        help="Blur threshold for frame selection. Higher values allow more blur. Lower values are stricter."
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

    if args.method == 'keyframes':
        check_ffmpeg()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.source.startswith(('http://', 'https://', 'www.')):
        # It's a YouTube URL
        video_path = f"downloaded_video_{timestamp}.mp4"
        video_title = download_video(args.source, video_path)
        sanitized_title = sanitize_filename(video_title)
    else:
        # It's a local file
        video_path = args.source
        video_title = os.path.splitext(os.path.basename(video_path))[0]
        sanitized_title = sanitize_filename(video_title)
    
    if args.output:
        output_folder = sanitize_filename(args.output)
    else:
        output_folder = f"screenshots_{sanitized_title}_{timestamp}"
    
    print(f"Video source: {video_path}")
    print(f"Extracting frames to: {output_folder}")
    print(f"Extraction method: {args.method}")
    print(f"Quality threshold set to {args.quality:.1f} (Range: 0-100, Higher is stricter)")
    print(f"Blur threshold set to {args.blur:.1f} (Higher values allow more blur)")
    
    extract_frames(video_path, output_folder, args.method, args.interval, args.quality, args.blur)
    print("Frame extraction complete.")

if __name__ == "__main__":
    main()
