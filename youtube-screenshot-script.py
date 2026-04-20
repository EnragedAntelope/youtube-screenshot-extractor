import cv2
import numpy as np
import os
import argparse
from datetime import datetime
import re
from PIL import Image
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from tqdm import tqdm
import time
import tempfile


def _get_subprocess_flags():
    """Get subprocess flags to hide console windows on Windows."""
    if sys.platform == "win32":
        return subprocess.CREATE_NO_WINDOW
    return 0


# Patch subprocess to hide console windows on Windows before importing yt_dlp
# This prevents FFmpeg popups when yt-dlp calls it internally for merging
if sys.platform == "win32":
    _original_popen_init = subprocess.Popen.__init__

    def _patched_popen_init(self, *args, **kwargs):
        if 'creationflags' not in kwargs:
            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
        _original_popen_init(self, *args, **kwargs)

    subprocess.Popen.__init__ = _patched_popen_init

import yt_dlp


def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                      creationflags=_get_subprocess_flags())
        return True
    except FileNotFoundError:
        return False

def sanitize_filename(filename):
    """Sanitize a filename by replacing unsafe characters with underscores."""
    return re.sub(r'[^\w\-_.]', '_', filename)


def clean_youtube_url(url):
    """Extract just the video ID from YouTube URLs, stripping playlist and other params.
    
    This prevents yt-dlp from trying to download entire playlists when the user
    only wants a single video. YouTube URLs often include ?list=PLAYLIST_ID which
    causes yt-dlp to iterate through all videos in the playlist.
    
    Args:
        url: YouTube URL (may contain playlist parameters)
        
    Returns:
        Clean URL with only the video ID
        
    Examples:
        >>> clean_youtube_url('https://youtu.be/VIDEO_ID?list=PLAYLIST')
        'https://youtu.be/VIDEO_ID'
        >>> clean_youtube_url('https://youtube.com/watch?v=VIDEO_ID&list=PLAYLIST')
        'https://youtube.com/watch?v=VIDEO_ID'
    """
    if not url or not isinstance(url, str):
        return url
    
    # Handle youtu.be short URLs
    if 'youtu.be' in url:
        # Extract video ID from path (everything after youtu.be/)
        match = re.search(r'youtu\.be/([a-zA-Z0-9_-]{11})', url)
        if match:
            video_id = match.group(1)
            return f'https://youtu.be/{video_id}'
    
    # Handle standard youtube.com/watch URLs
    if 'youtube.com' in url or 'youtube.com' in url:
        # Extract video ID from v= parameter
        match = re.search(r'[?&]v=([a-zA-Z0-9_-]{11})', url)
        if match:
            video_id = match.group(1)
            return f'https://www.youtube.com/watch?v={video_id}'
    
    # Handle youtube.com/embed URLs
    if 'youtube.com/embed' in url:
        match = re.search(r'youtube\.com/embed/([a-zA-Z0-9_-]{11})', url)
        if match:
            video_id = match.group(1)
            return f'https://www.youtube.com/watch?v={video_id}'
    
    # Return original if we couldn't parse it
    return url


def sanitize_output_path(path):
    """Sanitize an output path while preserving directory structure.

    If the path contains directory separators or is absolute, treat it as a
    user-specified path and preserve it (only sanitizing the final component).
    If it's just a simple name, sanitize the entire thing as a folder name.

    Args:
        path: User-provided output path (can be absolute, relative, or just a name)

    Returns:
        A safe path string that preserves the user's intended directory structure
    """
    if not path:
        return path

    # Normalize path separators for the current OS
    normalized = os.path.normpath(path)

    # Check if this is a path (has directory components) or just a folder name
    # A path has separators, or is absolute, or has a drive letter (Windows)
    has_separators = os.path.sep in path or '/' in path or '\\' in path
    is_absolute = os.path.isabs(normalized)
    has_drive = os.name == 'nt' and len(normalized) >= 2 and normalized[1] == ':'

    if has_separators or is_absolute or has_drive:
        # User specified a path - preserve directory structure
        # Split into directory and final component
        parent_dir = os.path.dirname(normalized)
        folder_name = os.path.basename(normalized)

        # Sanitize only the final folder name if it exists and isn't empty
        if folder_name:
            sanitized_name = sanitize_filename(folder_name)
            if parent_dir:
                return os.path.join(parent_dir, sanitized_name)
            else:
                return sanitized_name
        else:
            # Path ends with separator, use as-is (e.g., "C:\Screenshots\")
            return normalized
    else:
        # Just a folder name - sanitize the whole thing
        return sanitize_filename(path)

def safe_print(text):
    """Print text safely, handling Unicode characters on Windows console."""
    try:
        print(text)
    except UnicodeEncodeError:
        # Replace problematic characters with ASCII equivalents
        print(text.encode('ascii', 'replace').decode('ascii'))


def download_video(url, output_path, max_resolution=None, verbose=False, cookies_from_browser=None, cookies_file=None, sleep_requests=0, extractor_args=None):
    # Build format string with fallbacks for better compatibility
    if max_resolution:
        # Try requested resolution, fall back to best available if not found
        format_str = (
            f'bestvideo[height<={max_resolution}]+bestaudio/best[height<={max_resolution}]/'
            f'bestvideo+bestaudio/best'
        )
    else:
        format_str = 'bestvideo+bestaudio/best'

    ydl_opts = {
        'outtmpl': output_path,
        'format': format_str,
        'merge_output_format': 'mp4',
        'quiet': not verbose,
        'no_warnings': not verbose,
        'progress': verbose,
    }

    # Add cookie authentication if provided
    if cookies_from_browser:
        ydl_opts['cookiesfrombrowser'] = (cookies_from_browser,)
        if verbose:
            safe_print(f"Using cookies from browser: {cookies_from_browser}")
    elif cookies_file:
        ydl_opts['cookies'] = cookies_file
        if verbose:
            safe_print(f"Using cookies from file: {cookies_file}")

    # Add rate limiting to avoid bans
    if sleep_requests > 0:
        ydl_opts['sleep_requests'] = sleep_requests
        if verbose:
            safe_print(f"Rate limiting enabled: {sleep_requests} seconds between requests")

    # Add extractor arguments (e.g., for PO Tokens)
    if extractor_args:
        ydl_opts['extractor_args'] = extractor_args
        if verbose:
            safe_print(f"Using extractor arguments: {extractor_args}")

    if not check_ffmpeg():
        safe_print("Warning: FFmpeg is not installed. Downloading video only without merging audio.")
        ydl_opts['format'] = 'bestvideo/best'
        ydl_opts['postprocessors'] = []

    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_title = info.get('title', 'Unknown')
                if not verbose:
                    safe_print(f"Downloading: {video_title}...")
                ydl.download([url])
                if not verbose:
                    safe_print("Download complete.")
            return video_title
        except yt_dlp.utils.DownloadError as e:
            last_error = e
            error_msg = str(e)
            # Check for specific error types and give helpful messages
            if 'Requested format is not available' in error_msg:
                safe_print(f"Error: The requested video format is not available from this site.")
                safe_print("This site may not support the selected resolution or format.")
                safe_print("Try: Remove the resolution limit, or use a different source.")
                raise
            elif '403' in error_msg or 'Forbidden' in error_msg:
                if attempt < max_retries - 1:
                    safe_print(f"Download attempt {attempt + 1} failed (rate limited). Retrying...")
                    # Add exponential backoff
                    import time
                    time.sleep(2 ** attempt)
                    continue
                else:
                    safe_print("\n" + "="*70)
                    safe_print("ERROR: YouTube is blocking this request (HTTP 403)")
                    safe_print("="*70)
                    safe_print("\nThis usually means one of the following:")
                    safe_print("1. YouTube requires a PO Token for this video")
                    safe_print("2. Your IP has been rate limited")
                    safe_print("3. The video requires authentication (age-restricted, private, etc.)")
                    safe_print("\nSolutions:")
                    safe_print("- For PO Token issues: Use --cookies-from-browser firefox/chrome")
                    safe_print("- For rate limiting: Use --sleep-requests 5 (adds delay between requests)")
                    safe_print("- For private videos: Use --cookies-from-browser with an authenticated browser")
                    safe_print("- Update yt-dlp: pip install --upgrade 'yt-dlp[default]'")
                    safe_print("="*70 + "\n")
                    raise
            elif 'Private video' in error_msg or 'Sign in' in error_msg:
                safe_print("Error: This video is private or requires authentication.")
                safe_print("Use --cookies-from-browser firefox (or chrome) to access private videos.")
                raise
            elif any(k in error_msg for k in ('jsinterp', 'JavaScript', 'js interpreter', 'Deno', 'deno', 'external player')):
                safe_print("Error: A JavaScript runtime is required for YouTube downloads.")
                safe_print("Install Deno and ensure it is in your PATH:")
                safe_print("  Windows: winget install DenoLand.Deno  (or use START.bat option 3)")
                safe_print("  macOS:   brew install deno             (or use start.sh option 3)")
                safe_print("  Linux:   curl -fsSL https://deno.land/install.sh | sh")
                safe_print("Also ensure yt-dlp is up to date: pip install --upgrade \"yt-dlp[default]\"")
                raise
            elif 'po_token' in error_msg.lower() or 'po token' in error_msg.lower():
                safe_print("\n" + "="*70)
                safe_print("ERROR: This video requires a PO Token")
                safe_print("="*70)
                safe_print("\nYouTube now requires PO Tokens for many videos.")
                safe_print("To fix this, use browser cookies:")
                safe_print("  --cookies-from-browser firefox")
                safe_print("  or")
                safe_print("  --cookies-from-browser chrome")
                safe_print("\nMake sure you are logged into YouTube in that browser.")
                safe_print("="*70 + "\n")
                raise
            else:
                if attempt < max_retries - 1:
                    safe_print(f"Download attempt {attempt + 1} failed. Retrying...")
                    continue
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                safe_print(f"Download attempt {attempt + 1} failed. Retrying...")
            else:
                safe_print(f"Failed to download video after {max_retries} attempts.")
                raise

    safe_print(f"Failed to download video after {max_retries} attempts.")
    if last_error:
        raise last_error

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
        if h == 0 or w == 0:
            continue
        aspect_ratio = float(w) / h
        fill_ratio = cv2.contourArea(contour) / (w * h)
        
        if 0.5 < aspect_ratio < 2 and fill_ratio > threshold:
            if (x < frame.shape[1] * 0.2 and y < frame.shape[0] * 0.2) or \
               (x > frame.shape[1] * 0.8 and y < frame.shape[0] * 0.2) or \
               (x < frame.shape[1] * 0.2 and y > frame.shape[0] * 0.8) or \
               (x > frame.shape[1] * 0.8 and y > frame.shape[0] * 0.8):
                return True
    
    return False

def apply_filters(frame, gradfun, deblock, deband, verbose):
    filters_failed = []

    if gradfun:
        original_frame = frame.copy()
        frame = apply_ffmpeg_filter(frame, 'gradfun=1.2:8', verbose)
        if np.array_equal(original_frame, frame):
            filters_failed.append('gradfun')

    if deblock:
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    if deband:
        original_frame = frame.copy()
        frame = apply_ffmpeg_filter(frame, 'deband', verbose)
        if np.array_equal(original_frame, frame):
            filters_failed.append('deband')

    return frame, filters_failed


def apply_ffmpeg_filter(frame, filter_string, verbose):
    temp_in_name = None
    temp_out_name = None
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_in:
            temp_in_name = temp_in.name
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_out:
            temp_out_name = temp_out.name

        cv2.imwrite(temp_in_name, frame)
        ffmpeg_cmd = [
            'ffmpeg', '-i', temp_in_name, '-vf', filter_string, '-y', temp_out_name
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True,
                          creationflags=_get_subprocess_flags())
            return cv2.imread(temp_out_name)
        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"Error running FFmpeg command: {' '.join(map(shlex.quote, ffmpeg_cmd))}")
                print(f"Error output: {e.stderr}")
            return frame
        except FileNotFoundError:
            if verbose:
                print(f"Error: FFmpeg not found. Command attempted: {' '.join(map(shlex.quote, ffmpeg_cmd))}")
                print("Please ensure FFmpeg is installed and in your system PATH.")
            return frame
    finally:
        # Clean up temporary files
        if temp_in_name and os.path.exists(temp_in_name):
            os.unlink(temp_in_name)
        if temp_out_name and os.path.exists(temp_out_name):
            os.unlink(temp_out_name)

def process_frame(args):
    frame, output_folder, count, quality_threshold, blur_threshold, detect_watermarks, watermark_threshold, use_png, use_gpu, gradfun, deblock, deband, verbose = args
    
    if use_gpu:
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            from pycuda.compiler import SourceModule

            # GPU processing code
            cuda_code = """
            __global__ void process_image(unsigned char *d_image, int width, int height)
            {
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                int idy = threadIdx.y + blockIdx.y * blockDim.y;
                if (idx < width && idy < height)
                {
                    int offset = (idy * width + idx) * 3;
                    for (int i = 0; i < 3; i++)
                    {
                        float pixel_value = d_image[offset + i];
                        pixel_value = min(255.0f, pixel_value * 1.2f);
                        d_image[offset + i] = (unsigned char)pixel_value;
                    }
                }
            }
            """
            mod = SourceModule(cuda_code)
            process_image = mod.get_function("process_image")
            
            d_frame = cuda.mem_alloc(frame.nbytes)
            cuda.memcpy_htod(d_frame, frame)
            process_image(
                d_frame,
                np.int32(frame.shape[1]),
                np.int32(frame.shape[0]),
                block=(16, 16, 1),
                grid=((frame.shape[1] + 15) // 16, (frame.shape[0] + 15) // 16)
            )
            cuda.memcpy_dtoh(frame, d_frame)
        except Exception as e:
            if verbose:
                print(f"GPU processing failed: {e}. Falling back to CPU processing.")
            use_gpu = False
    
    if not use_gpu:
        # CPU processing (original processing logic)
        pass  # Your original CPU processing code goes here
    
    quality_score = calculate_quality_score(frame)
    laplacian_var = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    
    quality_check = quality_score >= quality_threshold
    blur_check = laplacian_var >= blur_threshold
    watermark_detected = detect_watermarks and detect_watermark(frame, watermark_threshold)
    
    if quality_check and blur_check:
        filters_failed = []
        if gradfun or deblock or deband:
            frame, filters_failed = apply_filters(frame, gradfun, deblock, deband, verbose)
        
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
        if filters_failed:
            status += f" (Filters failed: {', '.join(filters_failed)} - FFmpeg may not be installed)"
        return status, True, filters_failed
    else:
        skip_reason = []
        if not quality_check:
            skip_reason.append("low quality")
        if not blur_check:
            skip_reason.append("too blurry")
        return f"Skipped frame {count} due to: {' and '.join(skip_reason)} (Quality: {quality_score:.2f}, Blur: {laplacian_var:.2f})", False, []

def extract_frames(video_path, output_folder, method='interval', interval_seconds=5, quality_threshold=12, blur_threshold=10, detect_watermarks=False, watermark_threshold=0.8, use_parallel=True, use_png=False, use_gpu=False, fast_scene=False, resume=False, verbose=False, gradfun=False, deblock=False, deband=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Unable to open video file: {video_path}")
        sys.exit(1)
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if method == 'interval':
        frame_interval = int(fps * interval_seconds)
    elif method == 'all':
        frame_interval = 1
    elif method == 'keyframes':
        check_ffmpeg()
        output_pattern = os.path.join(output_folder, "keyframe_%03d.jpg")
        ffmpeg_command = [
            "ffmpeg", "-i", video_path,
            "-vf", "select='eq(pict_type,PICT_TYPE_I)',scale=in_range=full:out_range=tv,zscale=t=linear:npl=100:m=bt709:r=tv,format=yuv420p",
            "-fps_mode", "vfr",
            "-q:v", "2",
            output_pattern
        ]
        try:
            subprocess.run(ffmpeg_command, check=True, capture_output=True,
                          creationflags=_get_subprocess_flags())
        except subprocess.CalledProcessError as e:
            print(f"Error during keyframe extraction: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            video.release()
            return 0, 0, 0, set()
        video.release()
        print("Keyframe extraction complete.")
        return total_frames, 0, total_frames, set()
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
                    frames_to_process.append((frame, output_folder, i, quality_threshold, blur_threshold, detect_watermarks, watermark_threshold, use_png, use_gpu, gradfun, deblock, deband, verbose))
            
            video.release()
            
    if method != 'scene' or 'frames_to_process' not in locals():
        frames_to_process = []
        for frame_number in range(0, total_frames, frame_interval):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = video.read()
            if ret:
                frames_to_process.append((frame, output_folder, frame_number // frame_interval, quality_threshold, blur_threshold, detect_watermarks, watermark_threshold, use_png, use_gpu, gradfun, deblock, deband, verbose))
        
        video.release()
    
    skipped_frames = 0
    saved_frames = 0
    all_filters_failed = set()  # Track which filters failed across all frames

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
                    result, saved, filters_failed = future.result()
                    safe_print(result)
                    if saved:
                        saved_frames += 1
                    else:
                        skipped_frames += 1
                    all_filters_failed.update(filters_failed)
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
                result, saved, filters_failed = process_frame(args)
                safe_print(result)
                if saved:
                    saved_frames += 1
                else:
                    skipped_frames += 1
                all_filters_failed.update(filters_failed)
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

    return len(frames_to_process), skipped_frames, saved_frames, all_filters_failed

def generate_thumbnail(output_folder):
    frames = [f for f in os.listdir(output_folder) if f.endswith('.jpg') or f.endswith('.png')]
    # Exclude the montage itself if regenerating
    frames = [f for f in frames if not f.startswith('thumbnail_montage')]
    if not frames:
        print("No frames found to generate thumbnail.")
        return

    frames.sort()
    total = len(frames)

    # Select 9 frames spread across the output for variety
    # If we have fewer than 9 frames, just use what we have
    if total <= 9:
        selected_frames = frames
    else:
        # Pick frames at roughly equal intervals across the entire set
        # This gives a representative sample even with 1000+ frames
        indices = [int(i * (total - 1) / 8) for i in range(9)]
        selected_frames = [frames[i] for i in indices]

    images = [Image.open(os.path.join(output_folder, f)) for f in selected_frames]

    width, height = images[0].size
    # Calculate grid size based on number of images
    cols = min(3, len(images))
    rows = (len(images) + cols - 1) // cols
    thumbnail = Image.new('RGB', (width * cols, height * rows))

    for i, image in enumerate(images):
        thumbnail.paste(image, ((i % cols) * width, (i // cols) * height))

    thumbnail.save(os.path.join(output_folder, 'thumbnail_montage.jpg'))
    print(f"Thumbnail montage generated from {len(selected_frames)} frames (of {total} total).")

def main():
    if not check_ffmpeg():
        print("Warning: FFmpeg is not installed. Some features may be limited.")
        print("For full functionality, please install FFmpeg:")
        print("https://ffmpeg.org/download.html")
    
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

   5. Extract from age-restricted video using browser cookies:
      python youtube-screenshot-script.py "URL" --cookies-from-browser firefox

   6. Avoid rate limiting when processing multiple videos:
      python youtube-screenshot-script.py "URL" --sleep-requests 5

Post-processing Filters:
   --gradfun: Apply gradfun filter to reduce color banding (less aggressive, preserves more detail)
   --deblock: Apply deblocking filter to reduce compression artifacts
   --deband: Apply debanding filter to reduce color banding (more aggressive, better for severe banding)

YouTube Authentication (for age-restricted, private, or PO Token-required videos):
   --cookies-from-browser BROWSER: Use cookies from browser (firefox, chrome, edge, etc.)
   --cookies FILE: Use cookies from Netscape format file
   --sleep-requests SECONDS: Add delay between requests to avoid rate limiting
   --extractor-args ARGS: Additional yt-dlp extractor arguments (e.g., 'youtube:player_client=mweb')

Note: 
- Using filters may significantly increase processing time.
- Choose between gradfun and deband based on your needs:
   - Use gradfun for subtle banding issues or to preserve more detail.
   - Use deband for more severe banding problems, especially in dark scenes or sky gradients.
- YouTube now requires PO Tokens for many videos. Use --cookies-from-browser for best results.
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
    parser.add_argument("--gradfun", action="store_true", help="Apply gradfun filter to reduce color banding (less aggressive, preserves more detail)")
    parser.add_argument("--deblock", action="store_true", help="Apply deblocking filter")
    parser.add_argument("--deband", action="store_true", help="Apply debanding filter to reduce color banding (more aggressive, better for severe banding)")
    
    # YouTube authentication and rate limiting options
    parser.add_argument("--cookies-from-browser", type=str, metavar='BROWSER',
                        help="Load cookies from a browser. Use 'firefox' or 'chrome'. Required for age-restricted videos and helps with PO Token issues.")
    parser.add_argument("--cookies", type=str, metavar='FILE',
                        help="Path to a cookies file (Netscape format) for YouTube authentication")
    parser.add_argument("--sleep-requests", type=int, default=0, metavar='SECONDS',
                        help="Add a delay (in seconds) between requests to avoid rate limiting. Recommended: 3-5 for multiple videos.")
    parser.add_argument("--extractor-args", type=str, metavar='ARGS',
                        help="Additional extractor arguments for yt-dlp (e.g., 'youtube:player_client=mweb'). See yt-dlp documentation.")

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
        if not check_ffmpeg():
            print("Error: The 'keyframes' method requires FFmpeg.")
            print("Please install FFmpeg from https://ffmpeg.org/download.html")
            print("Or use the startup script option 4 to install it.")
            sys.exit(1)

    if args.use_gpu:
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            print("GPU acceleration is available.")
        except ImportError:
            print("Warning: PyCUDA is not installed. GPU acceleration is not available.")
            print("To use GPU acceleration, please install PyCUDA:")
            print("pip install pycuda>=2022.1")
            print("Falling back to CPU processing.")
            args.use_gpu = False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.source.startswith(('http://', 'https://', 'www.')):
        video_path = f"downloaded_video_{timestamp}.mp4"
        
        # Clean URL - remove playlist params that cause wrong video extraction
        cleaned_url = clean_youtube_url(args.source)
        if cleaned_url != args.source:
            safe_print(f"Note: Stripped playlist parameters from URL")
            safe_print(f"  Original: {args.source}")
            safe_print(f"  Cleaned:  {cleaned_url}")
        
        # Parse extractor_args if provided
        extractor_args_dict = None
        if args.extractor_args:
            # Parse key=value pairs separated by semicolons
            extractor_args_dict = {}
            for pair in args.extractor_args.split(';'):
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    extractor_args_dict[key.strip()] = value.strip()
        video_title = download_video(
            cleaned_url, video_path, args.max_resolution, args.verbose,
            cookies_from_browser=args.cookies_from_browser,
            cookies_file=args.cookies,
            sleep_requests=args.sleep_requests,
            extractor_args=extractor_args_dict
        )
        sanitized_title = sanitize_filename(video_title)
    else:
        # It's a local file
        video_path = args.source
        video_title = os.path.splitext(os.path.basename(video_path))[0]
        sanitized_title = sanitize_filename(video_title)
    
    if args.output:
        output_folder = sanitize_output_path(args.output)
    else:
        output_folder = f"screenshots_{sanitized_title}_{timestamp}"
    
    safe_print(f"Video source: {video_path}")
    safe_print(f"Extracting frames to: {output_folder}")
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
    if args.gradfun or args.deblock or args.deband:
        print("Post-processing filters enabled:")
        if args.gradfun:
            print("  - Gradfun filter")
        if args.deblock:
            print("  - Deblocking filter")
        if args.deband:
            print("  - Debanding filter")
    
    # Print authentication/rate limiting info
    if args.cookies_from_browser:
        print(f"Using cookies from browser: {args.cookies_from_browser}")
    elif args.cookies:
        print(f"Using cookies from file: {args.cookies}")
    if args.sleep_requests > 0:
        print(f"Rate limiting: {args.sleep_requests}s delay between requests")
    if args.extractor_args:
        print(f"Extractor arguments: {args.extractor_args}")
    
    if not args.dry_run:
        start_time = time.time()
        total_frames, skipped_frames, saved_frames, filters_failed = extract_frames(
            video_path, output_folder, args.method, args.interval, args.quality,
            args.blur, args.detect_watermarks, args.watermark_threshold,
            not args.disable_parallel, args.png, args.use_gpu, args.fast_scene,
            args.resume, args.verbose, args.gradfun, args.deblock, args.deband
        )
        end_time = time.time()

        execution_time = end_time - start_time
        frames_per_second = total_frames / execution_time if execution_time > 0 else 0

        print(f"\nFrame extraction complete.")
        print(f"Total execution time: {execution_time:.2f} seconds")
        print(f"Processed {total_frames} frames.")
        print(f"{saved_frames} high-quality frames saved!")
        print(f"{skipped_frames} frames skipped due to low-quality and/or blur.")
        print(f"Processing speed: {frames_per_second:.2f} frames/second")

        # Add information about post-processing filters
        if args.gradfun or args.deblock or args.deband:
            print("Post-processing filters:")
            if args.gradfun:
                if 'gradfun' in filters_failed:
                    print("  - Gradfun filter: FAILED (FFmpeg not found or filter error)")
                else:
                    print("  - Gradfun filter: applied successfully")
            if args.deblock:
                print("  - Deblocking filter: applied successfully")
            if args.deband:
                if 'deband' in filters_failed:
                    print("  - Debanding filter: FAILED (FFmpeg not found or filter error)")
                else:
                    print("  - Debanding filter: applied successfully")

            if filters_failed:
                print("\nWARNING: Some FFmpeg-based filters could not be applied.")
                print("Install FFmpeg for full filter support: https://ffmpeg.org/download.html")
                print("Or use the startup script option 4 to install it.")
        
        if args.thumbnail:
            generate_thumbnail(output_folder)
    else:
        print("Dry run completed. No frames were actually processed.")

if __name__ == "__main__":
    main()
