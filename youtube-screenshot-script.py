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
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from tqdm import tqdm
import time

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

def download_video(url, output_path, max_resolution=None, verbose=False):
    ydl_opts = {
        'outtmpl': output_path,
        'format': 'bestvideo+bestaudio/best',
        'merge_output_format': 'mp4',
        'verbose': verbose,
    }
    
    if max_resolution:
        ydl_opts['format'] = f'bestvideo[height<={max_resolution}]+bestaudio/best[height<={max_resolution}]'
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_title = info['title']
                ydl.download([url])
            return video_title
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Download attempt {attempt + 1} failed. Retrying...")
            else:
                print(f"Failed to download video after {max_retries} attempts.")
                raise e

def calculate_quality_score(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Sharpness using Laplacian variance
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = np.var(laplacian)
    sharpness_norm = min(max(sharpness / 1000, 0), 1.0)
    
    # Edge strength using Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edge_strength = np.mean(np.sqrt(sobelx**2 + sobely**2))
    edge_strength_norm = min(max(edge_strength / 100, 0), 1.0)
    
    # Contrast and Brightness
    contrast = np.std(gray) / (np.mean(gray) + 1e-6)  # Add small epsilon to avoid division by zero
    brightness = np.mean(gray) / 255
    
    # Advanced metrics
    entropy = cv2.calcHist([gray], [0], None, [256], [0, 256])
    entropy = entropy / (np.sum(entropy) + 1e-6)  # Normalize and avoid division by zero
    entropy = -np.sum(entropy * np.log2(entropy + 1e-7))
    entropy_norm = min(max(entropy / 8, 0), 1.0)  # 8 is max entropy for 8-bit image
    
    # Calculate weighted score
    score = (sharpness_norm * 0.3 + edge_strength_norm * 0.2 + contrast * 0.2 + brightness * 0.1 + entropy_norm * 0.2) * 100
    return max(min(score, 100), 0)  # Ensure the score is between 0 and 100

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

def detect_watermark(frame, threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    kernel = np.ones((5,5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        fill_ratio = cv2.contourArea(contour) / (w * h)
        
        if 0.5 < aspect_ratio < 2 and fill_ratio > threshold:
            if (x < frame.shape[1] * 0.2 and y < frame.shape[0] * 0.2) or \
               (x > frame.shape[1] * 0.8 and y < frame.shape[0] * 0.2) or \
               (x < frame.shape[1] * 0.2 and y > frame.shape[0] * 0.8) or \
               (x > frame.shape[1] * 0.8 and y > frame.shape[0] * 0.8):
                return True
    
    return False

def process_frame(args):
    frame, output_folder, count, quality_threshold, blur_threshold, detect_watermarks, watermark_threshold, use_png, use_gpu = args
    
    if use_gpu:
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            from pycuda.compiler import SourceModule
            # GPU processing code (unchanged)
            # ...
        except ImportError:
            print("Warning: pycuda is not installed. Falling back to CPU processing.")
            use_gpu = False
    
    quality_score = calculate_quality_score(frame)
    laplacian_var = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    
    quality_check = quality_score >= quality_threshold
    blur_check = laplacian_var >= blur_threshold
    watermark_detected = detect_watermarks and detect_watermark(frame, watermark_threshold)
    
    if quality_check and blur_check:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        pil_image = remove_black_bars(pil_image)
        
        filename = f"frame_{count:06d}_q{int(quality_score):02d}_b{int(laplacian_var):02d}"
        if watermark_detected:
            filename += "_watermarked"
        filename += ".png" if use_png else ".jpg"
        
        frame_filename = os.path.join(output_folder, filename)
        cv2.imwrite(frame_filename, cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR))
        
        status = f"Saved frame {frame_filename}"
        if watermark_detected:
            status += " (Watermark detected)"
        return status, True
    else:
        skip_reason = []
        if not quality_check:
            skip_reason.append("low quality")
        if not blur_check:
            skip_reason.append("too blurry")
        return f"Skipped frame {count} due to: {' and '.join(skip_reason)} (Quality: {quality_score:.2f}, Blur: {laplacian_var:.2f})", False

def extract_frames(video_path, output_folder, method='interval', interval_seconds=5, quality_threshold=12, blur_threshold=10, detect_watermarks=False, watermark_threshold=0.8, use_parallel=True, use_png=False, use_gpu=False, fast_scene=False, resume=False, verbose=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
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
        return total_frames, 0, total_frames
    elif method == 'scene':
        try:
            from scenedetect import detect, ContentDetector
        except ImportError:
            print("Error: The 'scene' method requires the scenedetect library.")
            print("Please install it manually using:")
            print("pip install scenedetect")
            sys.exit(1)
        
        try:
            scene_list = detect(video_path, ContentDetector(), fast_scene)
        except TypeError as e:
            print(f"Warning: Error during scene detection: {e}")
            print("Falling back to interval-based extraction.")
            method = 'interval'
            frame_interval = int(fps * interval_seconds)
        else:
            frames_to_process = []
            for i, scene in enumerate(scene_list):
                frame_number = scene[0].frame_num
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = video.read()
                if ret:
                    frames_to_process.append((frame, output_folder, i, quality_threshold, blur_threshold, detect_watermarks, watermark_threshold, use_png, use_gpu))
            
            video.release()
            
    if method != 'scene' or 'frames_to_process' not in locals():
        frames_to_process = []
        for frame_number in range(0, total_frames, frame_interval):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = video.read()
            if ret:
                frames_to_process.append((frame, output_folder, frame_number // frame_interval, quality_threshold, blur_threshold, detect_watermarks, watermark_threshold, use_png, use_gpu))
        
        video.release()
    
    skipped_frames = 0
    saved_frames = 0
    
    if resume:
        # Load progress from a file
        progress_file = os.path.join(output_folder, "progress.json")
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                progress = json.load(f)
            skipped_frames = progress["skipped_frames"]
            saved_frames = progress["saved_frames"]
            frames_to_process = frames_to_process[progress["processed_frames"]:]
    
    with tqdm(total=len(frames_to_process), disable=not verbose) as pbar:
        if use_parallel:
            with ThreadPoolExecutor() as executor:
                future_to_frame = {executor.submit(process_frame, args): args for args in frames_to_process}
                for future in as_completed(future_to_frame):
                    result, saved = future.result()
                    print(result)
                    if saved:
                        saved_frames += 1
                    else:
                        skipped_frames += 1
                    pbar.update(1)
                    
                    # Save progress
                    if resume:
                        progress = {
                            "processed_frames": len(frames_to_process) - len(future_to_frame),
                            "skipped_frames": skipped_frames,
                            "saved_frames": saved_frames
                        }
                        with open(progress_file, "w") as f:
                            json.dump(progress, f)
        else:
            for args in frames_to_process:
                result, saved = process_frame(args)
                print(result)
                if saved:
                    saved_frames += 1
                else:
                    skipped_frames += 1
                pbar.update(1)
                
                # Save progress
                if resume:
                    progress = {
                        "processed_frames": frames_to_process.index(args) + 1,
                        "skipped_frames": skipped_frames,
                        "saved_frames": saved_frames
                    }
                    with open(progress_file, "w") as f:
                        json.dump(progress, f)
    
    return len(frames_to_process), skipped_frames, saved_frames

def generate_thumbnail(output_folder):
    frames = [f for f in os.listdir(output_folder) if f.endswith('.jpg') or f.endswith('.png')]
    if not frames:
        print("No frames found to generate thumbnail.")
        return
    
    frames.sort()
    images = [Image.open(os.path.join(output_folder, f)) for f in frames[:9]]  # Take first 9 frames
    
    width, height = images[0].size
    thumbnail = Image.new('RGB', (width * 3, height * 3))
    
    for i, image in enumerate(images):
        thumbnail.paste(image, ((i % 3) * width, (i // 3) * height))
    
    thumbnail.save(os.path.join(output_folder, 'thumbnail_montage.jpg'))
    print("Thumbnail montage generated.")

def main():
    parser = argparse.ArgumentParser(
        description="Extract high-quality screenshots from YouTube videos or local video files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Output Filename Syntax:
  frame_NNNNNN_qXX_bYY[_watermarked].(jpg|png)
  where:
    NNNNNN: Frame number (zero-padded to 6 digits)
    XX: Quality score (0-99, higher is better)
    YY: Blur score (higher numbers indicate less blur)
    _watermarked: Suffix added if a watermark is detected (when --detect-watermarks is used)
    jpg|png: File extension based on the chosen format

Example: frame_000001_q85_b120_watermarked.png
         This is frame 1, with a quality score of 85, blur score of 120, a detected watermark, saved as PNG.

Usage Examples:
  1. Extract frames every 5 seconds from a YouTube video:
     python youtube-screenshot-script.py https://www.youtube.com/watch?v=dQw4w9WgXcQ

  2. Extract keyframes from a local video file:
     python youtube-screenshot-script.py path/to/your/video.mp4 --method keyframes

  3. Use scene detection on a YouTube video with custom output folder:
     python youtube-screenshot-script.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --method scene --output my_scene_shots

  4. Download a YouTube video at 720p and extract frames:
     python youtube-screenshot-script.py https://www.youtube.com/watch?v=dQw4w9WgXcQ --max-resolution 720
        """
    )
    parser.add_argument("source", help="YouTube video URL or path to local video file")
    parser.add_argument("--method", choices=['interval', 'all', 'keyframes', 'scene'], default='interval',
                        help="Frame extraction method (default: interval)")
    parser.add_argument("--interval", type=float, default=5.0, 
                        help="Interval between frames in seconds (default: 5.0, only used with 'interval' method)")
    parser.add_argument("--quality", type=float, default=12.0, 
                        help="Quality threshold for frame selection (0-100, default: 12.0)")
    parser.add_argument("--blur", type=float, default=10.0, 
                        help="Blur threshold for frame selection (default: 10.0)")
    parser.add_argument("--detect-watermarks", action="store_true", 
                        help="Enable basic watermark detection")
    parser.add_argument("--watermark-threshold", type=float, default=0.8, 
                        help="Watermark detection sensitivity (0-1, default: 0.8)")
    parser.add_argument("--max-resolution", type=int, 
                    help="Maximum resolution for YouTube video download (e.g., 720, 1080). Ignored for local files.")
    parser.add_argument("--output", type=str, default=None, 
                        help="Custom output folder name")
    parser.add_argument("--png", action="store_true", 
                        help="Save frames as PNG instead of JPG")
    parser.add_argument("--disable-parallel", action="store_true", 
                        help="Disable parallel processing of frames")
    parser.add_argument("--use-gpu", action="store_true", 
                        help="Use GPU acceleration if available")
    parser.add_argument("--fast-scene", action="store_true", 
                        help="Use fast mode for scene detection (less accurate results)")
    parser.add_argument("--resume", action="store_true", 
                        help="Resume an interrupted extraction process")
    parser.add_argument("--thumbnail", action="store_true", 
                        help="Generate a thumbnail montage of extracted frames")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable detailed logging")
    parser.add_argument("--dry-run", action="store_true", 
                        help="Show what would be done without actually processing")
    parser.add_argument("--config", type=str, 
                    help="Load settings from a JSON configuration file")

    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        parser.set_defaults(**config)
        args = parser.parse_args()

    if args.quality < 0 or args.quality > 100:
        parser.error("Quality threshold must be between 0 and 100.")
    
    if args.blur < 0 or args.blur > 1000:
        parser.error("Blur threshold must be between 0 and 1000.")
    
    if args.watermark_threshold < 0 or args.watermark_threshold > 1:
        parser.error("Watermark threshold must be between 0 and 1.")
    
    if args.interval <= 0:
        parser.error("Interval must be greater than 0.")

    if args.method == 'keyframes':
        check_ffmpeg()

    if args.use_gpu:
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
        except ImportError:
            print("Error: PyCUDA is not installed. GPU acceleration is not available.")
            print("To use GPU acceleration, please install PyCUDA:")
            print("pip install pycuda>=2022.1")
            print("Falling back to CPU processing.")
            args.use_gpu = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.source.startswith(('http://', 'https://', 'www.')):
        video_path = f"downloaded_video_{timestamp}.mp4"
        video_title = download_video(args.source, video_path, args.max_resolution, args.verbose)
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
    print(f"Blur threshold set to {args.blur:.1f} (Higher values allow less blur)")
    if args.detect_watermarks:
        print(f"Watermark detection enabled with threshold {args.watermark_threshold:.2f}")
    if args.use_gpu:
        print("GPU acceleration enabled")
    if args.fast_scene:
        print("Fast scene detection mode enabled")
    if args.resume:
        print("Resuming previous extraction process")
    if args.thumbnail:
        print("Thumbnail generation enabled")
    if args.dry_run:
        print("Dry run mode: No actual processing will occur")
    
    if not args.dry_run:
        start_time = time.time()
        total_frames, skipped_frames, saved_frames = extract_frames(
            video_path, output_folder, args.method, args.interval, args.quality, 
            args.blur, args.detect_watermarks, args.watermark_threshold, 
            not args.disable_parallel, args.png, args.use_gpu, args.fast_scene, 
            args.resume, args.verbose
        )
        end_time = time.time()
        
        execution_time = end_time - start_time
        frames_per_second = total_frames / execution_time
        
        print(f"\nFrame extraction complete.")
        print(f"Total execution time: {execution_time:.2f} seconds")
        print(f"Processed {total_frames} frames.")
        print(f"{saved_frames} high-quality frames saved!")
        print(f"{skipped_frames} frames skipped due to low-quality and/or blur.")
        print(f"Processing speed: {frames_per_second:.2f} frames/second")
        
        if args.thumbnail:
            generate_thumbnail(output_folder)
    else:
        print("Dry run completed. No frames were actually processed.")

if __name__ == "__main__":
    main()
