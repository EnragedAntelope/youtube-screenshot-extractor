import cv2
import numpy as np
import os
import argparse
from datetime import datetime
import re
import glob
from PIL import Image
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
import json
from functools import lru_cache
from tqdm import tqdm
import time


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


@lru_cache(maxsize=1)
def check_ffmpeg():
    """Return True if a runnable ffmpeg is on PATH.

    Cached: this is called from several places per run and each miss spawns a
    process (noticeably slow on Windows). Any OSError - not just a missing
    binary - means we cannot use FFmpeg, so treat them all as "unavailable"
    rather than letting them abort the run.
    """
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                      creationflags=_get_subprocess_flags())
        return True
    except OSError:
        return False

def sanitize_filename(filename):
    """Sanitize a filename by replacing unsafe characters with underscores."""
    return re.sub(r'[^\w\-_.]', '_', filename)


def remove_partial_download(video_path):
    """Delete a partially downloaded video plus yt-dlp's bookkeeping files.

    yt-dlp streams into '<target>.part' (and fragment pieces) before the
    final file appears, so a failed or interrupted download leaves several
    hundred MB of orphans behind unless they are removed explicitly.
    """
    for leftover in glob.glob(glob.escape(video_path) + "*"):
        try:
            os.remove(leftover)
        except OSError as e:
            safe_print(f"Note: Could not remove partial download {leftover}: {e}")


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
        'https://www.youtube.com/watch?v=VIDEO_ID'
    """
    if not url or not isinstance(url, str):
        return url

    # Handle youtu.be short URLs
    if 'youtu.be' in url:
        match = re.search(r'youtu\.be/([a-zA-Z0-9_-]{11})', url)
        if match:
            return f'https://youtu.be/{match.group(1)}'

    if 'youtube.com' in url:
        # Standard watch URLs: extract video ID from v= parameter
        match = re.search(r'[?&]v=([a-zA-Z0-9_-]{11})', url)
        if match:
            return f'https://www.youtube.com/watch?v={match.group(1)}'

        # Shorts, live, and embed URLs
        match = re.search(r'youtube\.com/(?:shorts|live|embed)/([a-zA-Z0-9_-]{11})', url)
        if match:
            return f'https://www.youtube.com/watch?v={match.group(1)}'

    # Return original if we couldn't parse it
    return url


def parse_extractor_args(values):
    """Parse --extractor-args strings into yt-dlp's nested option format.

    yt-dlp expects extractor_args as {ie_key: {arg_name: [values]}} - NOT a flat
    dict of strings. Its CLI grammar is 'IE_KEY:ARG1=VAL1,VAL2;ARG2=VAL3', and the
    flag may be repeated for multiple extractors.

    Args:
        values: list of raw --extractor-args strings (argparse 'append' action)

    Returns:
        Nested dict suitable for ydl_opts['extractor_args'], e.g.
        parse_extractor_args(['youtube:player_client=mweb'])
            -> {'youtube': {'player_client': ['mweb']}}

    Raises:
        ValueError: if a value is missing the required 'EXTRACTOR:' prefix.
    """
    result = {}
    for value in values:
        if ':' not in value:
            raise ValueError(
                f"Invalid --extractor-args '{value}'. "
                "Expected format: EXTRACTOR:ARG=VALUE (e.g. youtube:player_client=mweb)"
            )
        ie_key, args_str = value.split(':', 1)
        ie_args = result.setdefault(ie_key.strip().lower(), {})
        for item in args_str.split(';'):
            item = item.strip()
            if not item:
                continue
            if '=' in item:
                key, val = item.split('=', 1)
                ie_args[key.strip()] = [v.strip() for v in val.split(',')]
            else:
                # Bare flag with no value (rare) - yt-dlp treats these as empty lists
                ie_args[item] = []
    return result


def _describe_action_type(action):
    """Human-readable name for what an argparse action accepts."""
    if action.nargs == 0:
        return "true or false"
    if isinstance(action, argparse._AppendAction):
        return "a string or a list of strings"
    return {float: "a number", int: "a whole number", str: "a string"}.get(
        action.type, getattr(action.type, "__name__", "a value")
    )


def _coerce_config_value(parser, path, key, value, action):
    """Validate one config entry against the option it maps to."""
    expected = _describe_action_type(action)

    def bad():
        parser.error(f"Setting '{key}' in {path} must be {expected}, got {value!r}.")

    if action.nargs == 0:  # store_true / store_false
        if not isinstance(value, bool):
            bad()
        return value

    if isinstance(action, argparse._AppendAction):
        if isinstance(value, str):
            return [value]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            bad()
        return value

    # bool is a subclass of int, so float(True) succeeds and would silently turn
    # a mistyped flag into the number 1.
    if isinstance(value, bool):
        bad()

    if action.type is str and not isinstance(value, str):
        bad()

    # int(5.7) would truncate silently, while the CLI rejects --sleep-requests 5.7.
    if action.type is int and isinstance(value, float) and not value.is_integer():
        bad()

    if action.type is not None:
        try:
            value = action.type(value)
        except (TypeError, ValueError):
            bad()
    elif action.choices is not None and not isinstance(value, str):
        # Untyped options with choices (--method) are plain strings.
        bad()

    if action.choices is not None and value not in action.choices:
        choices = ", ".join(str(c) for c in action.choices)
        parser.error(f"Setting '{key}' in {path} must be one of: {choices}. Got {value!r}.")

    return value


def apply_config_file(parser, path):
    """Apply a JSON config file as argparse defaults, validating as we go.

    Previously this was a bare ``set_defaults(**json.load(f))``, which accepted
    anything: a typo became a silently ignored setting, and a wrong type
    surfaced much later as a traceback from inside the extraction. Check each
    entry against the parser's own option table instead, so the file is
    validated in one place with errors that name the offending key.

    Returns the settings for 'append' options, which the caller must apply
    after parsing - see the comment at the end of this function.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            config = json.load(f)
    except OSError as e:
        parser.error(f"Could not read config file {path}: {e}")
    except json.JSONDecodeError as e:
        parser.error(f"Config file {path} is not valid JSON: {e}")

    if not isinstance(config, dict):
        parser.error(f"Config file {path} must contain a JSON object of settings.")

    # argparse exposes no public option table, so read the actions directly.
    known = {a.dest: a for a in parser._actions if a.dest not in ("help", "config")}

    validated = {}
    for key, value in config.items():
        dest = key.lstrip("-").replace("-", "_")
        action = known.get(dest)
        if action is None:
            # Suggest only what a user could usefully put in a file: not the
            # positional source, and not options hidden from --help.
            valid = ", ".join(sorted(
                d for d, a in known.items()
                if d != "source" and a.help is not argparse.SUPPRESS
            ))
            parser.error(f"Unknown setting '{key}' in config file {path}. Valid settings: {valid}")
        if dest == "source":
            parser.error(
                f"'source' cannot be set from a config file ({path}) - it is a "
                "positional argument, so pass the URL or file path on the command line."
            )
        validated[dest] = _coerce_config_value(parser, path, key, value, action)

    # 'append' options are handled by the caller, not set_defaults: argparse's
    # append action EXTENDS a non-None default rather than replacing it, so a
    # value from the file would merge with one given on the command line
    # instead of being overridden by it like every other setting.
    appends = {d: v for d, v in validated.items()
               if isinstance(known[d], argparse._AppendAction)}
    parser.set_defaults(**{d: v for d, v in validated.items() if d not in appends})
    return appends


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


def build_ydl_opts(output_path, max_resolution=None, verbose=False, cookies_from_browser=None,
                   cookies_file=None, sleep_requests=0, extractor_args=None):
    """Build the yt-dlp options dict shared by download and dry-run info fetch."""
    if max_resolution:
        # Try requested resolution, fall back to best available if not found
        format_str = (
            f'bestvideo[height<={max_resolution}]+bestaudio/best[height<={max_resolution}]/'
            f'bestvideo+bestaudio/best'
        )
        # Video-only variant, used when FFmpeg is missing and streams cannot be
        # merged. It still has to honour the resolution cap - dropping it would
        # silently download the largest stream available.
        novideo_merge_format_str = (
            f'bestvideo[height<={max_resolution}]/best[height<={max_resolution}]/'
            f'bestvideo/best'
        )
    else:
        format_str = 'bestvideo+bestaudio/best'
        novideo_merge_format_str = 'bestvideo/best'

    ydl_opts = {
        'outtmpl': output_path,
        'format': format_str,
        'merge_output_format': 'mp4',
        'quiet': not verbose,
        'no_warnings': not verbose,
        'noprogress': not verbose,
    }

    # Add cookie authentication if provided
    if cookies_from_browser:
        ydl_opts['cookiesfrombrowser'] = (cookies_from_browser,)
        if verbose:
            safe_print(f"Using cookies from browser: {cookies_from_browser}")
    elif cookies_file:
        ydl_opts['cookiefile'] = cookies_file
        if verbose:
            safe_print(f"Using cookies from file: {cookies_file}")

    # Add rate limiting to avoid bans
    if sleep_requests > 0:
        ydl_opts['sleep_interval_requests'] = sleep_requests
        if verbose:
            safe_print(f"Rate limiting enabled: {sleep_requests} seconds between requests")

    # Add extractor arguments (e.g., for PO Tokens)
    if extractor_args:
        ydl_opts['extractor_args'] = extractor_args
        if verbose:
            safe_print(f"Using extractor arguments: {extractor_args}")

    if not check_ffmpeg():
        safe_print("Warning: FFmpeg is not installed. Downloading video only without merging audio.")
        ydl_opts['format'] = novideo_merge_format_str
        ydl_opts['postprocessors'] = []

    return ydl_opts


def get_video_info(url, **kwargs):
    """Fetch video metadata without downloading (used by --dry-run)."""
    ydl_opts = build_ydl_opts('%(title)s.%(ext)s', **kwargs)
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        return ydl.extract_info(url, download=False)


def download_video(url, output_path, max_resolution=None, verbose=False, cookies_from_browser=None, cookies_file=None, sleep_requests=0, extractor_args=None):
    ydl_opts = build_ydl_opts(
        output_path, max_resolution, verbose,
        cookies_from_browser, cookies_file, sleep_requests, extractor_args
    )

    max_retries = 3
    last_error = None
    for attempt in range(max_retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                if not verbose:
                    safe_print("Downloading...")
                # A single extract_info(download=True) does the metadata fetch and
                # the download in one pass. Fetching info separately first would
                # double the requests we make, which works against --sleep-requests
                # and YouTube's per-hour limits.
                info = ydl.extract_info(url, download=True)
                video_title = info.get('title', 'Unknown')
                if not verbose:
                    safe_print(f"Download complete: {video_title}")
            return video_title
        except yt_dlp.utils.DownloadError as e:
            last_error = e
            error_msg = str(e)
            # Check for specific error types and give helpful messages
            if 'Requested format is not available' in error_msg:
                safe_print("Error: The requested video format is not available from this site.")
                safe_print("This site may not support the selected resolution or format.")
                safe_print("Try: Remove the resolution limit, or use a different source.")
                raise
            elif '403' in error_msg or 'Forbidden' in error_msg:
                if attempt < max_retries - 1:
                    safe_print(f"Download attempt {attempt + 1} failed (rate limited). Retrying...")
                    # Add exponential backoff
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
                    time.sleep(2 ** attempt)
                    continue
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                safe_print(f"Download attempt {attempt + 1} failed. Retrying...")
                time.sleep(2 ** attempt)
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
    contrast_norm = min(max(contrast, 0), 1.0)  # Clamp like the other terms so the 0-100 scale stays uniform
    brightness = np.mean(gray) / 255

    # Advanced metrics
    entropy = cv2.calcHist([gray], [0], None, [256], [0, 256])
    entropy = entropy / (np.sum(entropy) + 1e-6)  # Normalize and avoid division by zero
    entropy = -np.sum(entropy * np.log2(entropy + 1e-7))
    entropy_norm = min(max(entropy / 8, 0), 1.0)  # 8 is max entropy for 8-bit image

    # Calculate weighted score
    score = (sharpness_norm * 0.3 + edge_strength_norm * 0.2 + contrast_norm * 0.2 + brightness * 0.1 + entropy_norm * 0.2) * 100
    return max(min(score, 100), 0)  # Ensure the score is between 0 and 100

def remove_black_bars(frame, threshold=10):
    """Crop black letterbox/pillarbox bars from a BGR frame.

    A row/column is considered a bar only if every pixel in it is darker
    than the threshold. Returns the frame unchanged if it is entirely black.
    """
    # Fast path. If all four outermost edges already contain something brighter
    # than the threshold there is no bar on any side, and the full scan below
    # would return the frame unchanged - so skip it. This matters because
    # cropping now runs on every frame (scores have to describe the cropped
    # image), the full scan costs about as much as the quality scoring itself,
    # and most video has no bars at all. O(W+H) instead of O(W*H*3).
    edges = frame[:, :, :3]
    if (edges[0].max() >= threshold and edges[-1].max() >= threshold
            and edges[:, 0].max() >= threshold and edges[:, -1].max() >= threshold):
        return frame

    black_mask = (frame[:, :, :3] < threshold).all(axis=2)

    content_rows = np.where(~black_mask.all(axis=1))[0]
    if content_rows.size == 0:
        return frame  # Entirely black frame - nothing to crop
    top, bottom = content_rows[0], content_rows[-1]

    content_cols = np.where(~black_mask[top:bottom + 1].all(axis=0))[0]
    if content_cols.size == 0:
        return frame
    left, right = content_cols[0], content_cols[-1]

    return frame[top:bottom + 1, left:right + 1]

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

def apply_filters(frame, gradfun, deblock, deband, verbose, ffmpeg_available=True):
    filters_failed = []

    # Deblock is pure OpenCV - no FFmpeg needed.
    if deblock:
        frame = cv2.fastNlMeansDenoisingColored(frame, None, 10, 10, 7, 21)

    # gradfun and deband both run through FFmpeg. Chain them into a single
    # invocation so we spawn at most one FFmpeg process per frame rather than one
    # per filter.
    ffmpeg_chain = []
    if gradfun:
        ffmpeg_chain.append('gradfun=1.2:8')
    if deband:
        ffmpeg_chain.append('deband')

    if ffmpeg_chain:
        requested = [name for name, on in (('gradfun', gradfun), ('deband', deband)) if on]
        if not ffmpeg_available:
            filters_failed.extend(requested)
        else:
            filtered = apply_ffmpeg_filter(frame, ','.join(ffmpeg_chain), verbose)
            if filtered is None:
                # FFmpeg failed; the chained filters are applied together, so
                # report all of them as failed.
                filters_failed.extend(requested)
            else:
                frame = filtered

    return frame, filters_failed


def apply_ffmpeg_filter(frame, filter_string, verbose):
    """Run an FFmpeg ``-vf`` filter chain on a single BGR frame.

    Frames are piped as raw video rather than round-tripped through a PNG on
    disk: gradfun and deband never change frame dimensions, so rawvideo in
    and out is lossless and skips an encode/decode cycle per frame. Returns
    the filtered frame, or ``None`` if FFmpeg failed (missing binary,
    non-zero exit, or short read) so callers can report the failure instead
    of silently keeping the unfiltered frame.
    """
    height, width = frame.shape[:2]
    ffmpeg_cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin',
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-video_size', f'{width}x{height}',
        '-i', 'pipe:0',
        '-vf', filter_string,
        '-f', 'rawvideo', '-pix_fmt', 'bgr24', 'pipe:1'
    ]
    try:
        result = subprocess.run(ffmpeg_cmd, input=frame.tobytes(), capture_output=True,
                                creationflags=_get_subprocess_flags())
    except FileNotFoundError:
        if verbose:
            print(f"Error: FFmpeg not found. Command attempted: {' '.join(map(shlex.quote, ffmpeg_cmd))}")
            print("Please ensure FFmpeg is installed and in your system PATH.")
        return None

    expected_bytes = width * height * 3
    if result.returncode != 0 or len(result.stdout) != expected_bytes:
        if verbose:
            print(f"Error running FFmpeg command: {' '.join(map(shlex.quote, ffmpeg_cmd))}")
            print(f"Error output: {result.stderr.decode(errors='replace')}")
        return None
    return np.frombuffer(result.stdout, np.uint8).reshape(height, width, 3)

def process_frame(args):
    (frame, output_folder, count, quality_threshold, blur_threshold, detect_watermarks,
     watermark_threshold, use_png, gradfun, deblock, deband, verbose, ffmpeg_available) = args

    # Crop before scoring. Letterbox/pillarbox bars are large flat black regions
    # that drag down contrast and entropy, so scoring the uncropped frame judged
    # the bars as much as the picture - and left the filename's scores describing
    # a frame that was never saved. Watermark detection needs the crop too: its
    # corner tests are relative to frame.shape, and the real corners of the
    # picture are the cropped ones.
    frame = remove_black_bars(frame)

    quality_score = calculate_quality_score(frame)
    laplacian_var = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

    quality_check = quality_score >= quality_threshold
    blur_check = laplacian_var >= blur_threshold

    if quality_check and blur_check:
        watermark_detected = detect_watermarks and detect_watermark(frame, watermark_threshold)

        filters_failed = []
        if gradfun or deblock or deband:
            # Applied after scoring on purpose: deblock is a denoiser, so it
            # lowers Laplacian variance by design. Scoring its output would
            # stamp a "blurrier" number on a frame the filter just improved.
            frame, filters_failed = apply_filters(frame, gradfun, deblock, deband, verbose, ffmpeg_available)

        filename = f"frame_{count:06d}_q{int(quality_score):02d}_b{int(laplacian_var):02d}"
        if watermark_detected:
            filename += "_watermarked"
        filename += ".png" if use_png else ".jpg"

        frame_filename = os.path.join(output_folder, filename)
        cv2.imwrite(frame_filename, frame)

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

def detect_scene_frames(video_path, fast_scene=False, verbose=False):
    """Run scene detection and return the list of scene-start frame numbers.

    Uses the SceneManager API directly so fast mode can skip frames during
    detection (roughly 3x faster, slightly less accurate cut placement).
    """
    try:
        from scenedetect import open_video, SceneManager, ContentDetector
    except ImportError:
        print("Error: The 'scene' method requires the scenedetect library.")
        print("Please install it manually using:")
        print("pip install scenedetect")
        sys.exit(1)

    video_stream = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    scene_manager.detect_scenes(
        video_stream,
        frame_skip=2 if fast_scene else 0,
        show_progress=verbose,
    )
    scene_list = scene_manager.get_scene_list()
    if not scene_list:
        # Video with no detected cuts (single continuous shot) - treat the
        # whole video as one scene so at least one frame is extracted.
        safe_print("No scene changes detected; treating the video as a single scene.")
        return [0]
    return [scene[0].frame_num for scene in scene_list]


class _StatusReporter:
    """Progress output tuned to where it is actually going.

    A terminal gets a tqdm bar. When stdout is a pipe it does not: the GUI runs
    this script as a subprocess and streams it into a Tk text widget, where a
    bar built from carriage returns renders as thousands of separate lines. That
    case gets a periodic one-line summary instead.

    The GUI additionally needs a machine-parsable feed for its determinate
    progress bar. It sets YSE_PROGRESS=1 in the child's environment; when
    present, every advance() also emits one @@PROGRESS line, which the GUI
    parses and strips from the log. Terminal runs never emit them.

    Per-frame lines are verbose-only either way. They used to print
    unconditionally while the bar was verbose-only, which is backwards - an
    'all'-method run over a few minutes of video emits tens of thousands of
    them, burying the summary that follows.
    """

    def __init__(self, total, initial, verbose, summary_interval=2.0):
        self.verbose = verbose
        self.is_tty = bool(getattr(sys.stdout, 'isatty', lambda: False)())
        self.summary_interval = summary_interval
        self._last_summary = 0.0
        # The GUI sets YSE_PROGRESS=1 and parses @@PROGRESS lines into its
        # determinate bar, stripping them from the log. Terminals never see
        # them, so human-facing output is unchanged.
        self._machine_progress = os.environ.get("YSE_PROGRESS") == "1"
        self._bar = tqdm(total=total, initial=initial, disable=not self.is_tty,
                         unit="frame")

    @staticmethod
    def _emit(text):
        # tqdm.write keeps the bar from being torn apart by interleaved output.
        try:
            tqdm.write(text)
        except UnicodeEncodeError:
            tqdm.write(text.encode('ascii', 'replace').decode('ascii'))

    def log(self, message):
        """Report one frame's outcome. Verbose only."""
        if self.verbose:
            self._emit(message)

    def advance(self, tracker):
        self._bar.update(1)
        if self._machine_progress:
            self._emit(self.progress_line(tracker))
        if self.is_tty or self.verbose:
            return
        now = time.monotonic()
        if now - self._last_summary < self.summary_interval:
            return
        self._last_summary = now
        self._emit(self.summary_line(tracker))

    def progress_line(self, tracker):
        """One machine-parsable snapshot for the GUI's progress bar."""
        parts = [f"done={tracker.processed}"]
        if self._bar.total:
            parts.append(f"total={self._bar.total}")
        parts.append(f"saved={tracker.saved} skipped={tracker.skipped}")
        return "@@PROGRESS " + " ".join(parts)

    def summary_line(self, tracker):
        total = self._bar.total
        position = f"{tracker.processed}/{total}" if total else str(tracker.processed)
        return (f"Progress: {position} frames - "
                f"{tracker.saved} saved, {tracker.skipped} skipped")

    def close(self, tracker=None):
        self._bar.close()
        # One final line for the piped case, which never saw the bar.
        if tracker is not None and not self.is_tty and not self.verbose:
            self._emit(self.summary_line(tracker))


def load_progress(progress_file):
    """Read a resume checkpoint, returning (processed, skipped, saved).

    The progress file is written by an extraction that may have been killed
    mid-write, and --resume exists precisely for those runs - so a truncated or
    unreadable file is an expected input, not an exotic one. Fall back to
    starting over rather than aborting the run with a traceback.
    """
    try:
        with open(progress_file, "r") as f:
            progress = json.load(f)
        return (
            int(progress.get("processed_frames", 0)),
            int(progress.get("skipped_frames", 0)),
            int(progress.get("saved_frames", 0)),
        )
    except (json.JSONDecodeError, OSError, TypeError, ValueError, AttributeError) as e:
        safe_print(f"Warning: could not read {progress_file} ({e}).")
        safe_print("Starting this extraction from the beginning.")
        return (0, 0, 0)


class _ProgressTracker:
    """Fold out-of-order frame completions into a contiguous prefix.

    Under parallel processing, frame tasks finish out of order. A plain "count of
    completed frames" is therefore NOT a safe resume point: frame N may still be
    in flight while later frames have finished, so skipping the first N frames on
    resume could drop frame N entirely.

    This tracker instead advances ``resume_point`` only across a contiguous run of
    completed frame indices, so every index below ``resume_point`` is guaranteed
    done. Saved/skipped tallies are folded in the same contiguous order, keeping
    the persisted counts consistent with the resume point (no double-counting of
    the frames re-processed after a resume).
    """

    def __init__(self, start_at=0, saved=0, skipped=0):
        self.resume_point = start_at
        self.saved = saved
        self.skipped = skipped
        self._pending = {}  # count -> was_saved, for done-but-not-yet-contiguous frames

    def record(self, count, was_saved):
        self._pending[count] = was_saved
        while self.resume_point in self._pending:
            if self._pending.pop(self.resume_point):
                self.saved += 1
            else:
                self.skipped += 1
            self.resume_point += 1

    @property
    def processed(self):
        # After a complete run every frame is contiguous, so this equals the
        # total number of frames reached.
        return self.resume_point


def extract_frames(video_path, output_folder, method='interval', interval_seconds=5, quality_threshold=30, blur_threshold=50, detect_watermarks=False, watermark_threshold=0.8, use_parallel=True, use_png=False, fast_scene=False, resume=False, verbose=False, gradfun=False, deblock=False, deband=False):
    os.makedirs(output_folder, exist_ok=True)

    if method == 'keyframes':
        extension = "png" if use_png else "jpg"
        # %06d rather than %03d: three digits wrap at frame 1000, which then
        # sorts lexically before keyframe_999. Six digits matches the
        # frame_%06d naming every other method uses.
        output_pattern = os.path.join(output_folder, f"keyframe_%06d.{extension}")
        # abspath, not the raw path. FFmpeg has no "--" end-of-options marker,
        # so a leading '-' in the name would be read as a flag; argparse
        # rejects such a source first, so this is belt-and-braces rather than
        # a live hole, but it also pins the input against any cwd change.
        ffmpeg_command = [
            "ffmpeg", "-hide_banner", "-nostdin", "-y", "-i", os.path.abspath(video_path),
            "-vf", "select='eq(pict_type,PICT_TYPE_I)'",
            "-fps_mode", "vfr",
            "-q:v", "2",
            output_pattern
        ]
        # Snapshot the folder so the count below can tell this extraction's
        # output from leftovers. Each file is compared against its OWN earlier
        # stat rather than against a wall-clock instant: two runs seconds apart
        # are indistinguishable by "mtime >= started" once filesystem timestamp
        # granularity is allowed for.
        keyframe_glob = os.path.join(output_folder, f"keyframe_*.{extension}")

        def stat_keyframes():
            snapshot = {}
            for path in glob.glob(keyframe_glob):
                try:
                    info = os.stat(path)
                except OSError:
                    continue
                snapshot[path] = (info.st_mtime_ns, info.st_size)
            return snapshot

        before = stat_keyframes()
        try:
            # -y and -nostdin: without them FFmpeg prompts "Overwrite? [y/N]" when
            # the output folder already holds keyframes from a previous run and
            # then blocks forever, because stdout/stderr are captured but stdin is
            # inherited (and there is no console at all under the GUI).
            subprocess.run(ffmpeg_command, check=True, capture_output=True,
                          creationflags=_get_subprocess_flags())
        except subprocess.CalledProcessError as e:
            print(f"Error during keyframe extraction: {e.stderr.decode() if e.stderr else 'Unknown error'}")
            return 0, 0, 0, set()
        # Count what this run wrote, not what is in the folder. FFmpeg numbers
        # from 1 every time, so re-using an --output folder overwrites the low
        # numbers and leaves any higher ones from a longer previous video in
        # place - counting the folder reported those as saved by this run too.
        after = stat_keyframes()
        saved = sum(1 for path, info in after.items() if before.get(path) != info)
        stale = len(after) - saved
        print("Keyframe extraction complete.")
        if stale:
            safe_print(f"Note: {stale} keyframe file(s) already in {output_folder} "
                       "are from an earlier run and were left untouched.")
        return saved, 0, saved, set()

    scene_frame_numbers = None
    if method == 'scene':
        try:
            scene_frame_numbers = detect_scene_frames(video_path, fast_scene, verbose)
        except Exception as e:
            print(f"Warning: Error during scene detection: {e}")
            print("Falling back to interval-based extraction.")
            method = 'interval'

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Unable to open video file: {video_path}")
        sys.exit(1)

    fps = video.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        safe_print("Warning: Could not determine video FPS. Assuming 30 fps for interval calculation.")
        fps = 30.0
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_step = 1 if method == 'all' else max(1, int(round(fps * interval_seconds)))

    # Estimated number of frames that will be processed (for the progress bar).
    if scene_frame_numbers is not None:
        expected_total = len(scene_frame_numbers)
    elif total_frames > 0:
        expected_total = (total_frames + frame_step - 1) // frame_step
    else:
        expected_total = None  # Unknown length (e.g., some streams)

    def frame_generator():
        """Yield (frame, count) one at a time so memory stays bounded.

        ``count`` numbers the frames this generator actually yields, with no
        gaps. It is not an index into the source: _ProgressTracker only
        advances across a contiguous run of completed counts, so a skipped
        number would strand every later frame in its pending map - totals
        would under-report and --resume would checkpoint at the gap and
        re-process everything after it.
        """
        if scene_frame_numbers is not None:
            # Seeks can fail (short or damaged tail, variable frame rate), so
            # number the frames that actually decode rather than the scene list.
            count = 0
            for frame_number in scene_frame_numbers:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = video.read()
                if ret:
                    yield frame, count
                    count += 1
                elif verbose:
                    safe_print(f"Warning: could not read scene frame {frame_number}; skipping it.")
        else:
            # Sequential read with grab() to skip undecoded frames quickly
            frame_number = 0
            count = 0
            while True:
                ret = video.grab()
                if not ret:
                    break
                if frame_number % frame_step == 0:
                    ret, frame = video.retrieve()
                    if ret:
                        yield frame, count
                        count += 1
                frame_number += 1

    start_at = 0
    saved_frames = 0
    skipped_frames = 0
    all_filters_failed = set()  # Track which filters failed across all frames
    progress_file = os.path.join(output_folder, "progress.json")
    ffmpeg_available = check_ffmpeg()  # Checked once; passed to workers instead of per-frame

    if resume and os.path.exists(progress_file):
        start_at, skipped_frames, saved_frames = load_progress(progress_file)
        if start_at or skipped_frames or saved_frames:
            safe_print(f"Resuming: skipping first {start_at} already-processed frames.")

    tracker = _ProgressTracker(start_at, saved_frames, skipped_frames)

    def save_progress():
        if not resume:
            return
        # Write-then-replace: os.replace is atomic, so an interrupt can never
        # leave a half-written progress file behind for the next --resume run.
        tmp_path = progress_file + ".tmp"
        try:
            with open(tmp_path, "w") as f:
                json.dump({
                    "processed_frames": tracker.resume_point,
                    "skipped_frames": tracker.skipped,
                    "saved_frames": tracker.saved,
                }, f)
            os.replace(tmp_path, progress_file)
        except OSError as e:
            safe_print(f"Warning: could not save resume progress: {e}")

    def make_task_args(frame, count):
        return (frame, output_folder, count, quality_threshold, blur_threshold,
                detect_watermarks, watermark_threshold, use_png,
                gradfun, deblock, deband, verbose, ffmpeg_available)

    def handle_result(count, future_or_result):
        try:
            result, saved, filters_failed = (
                future_or_result.result() if hasattr(future_or_result, 'result') else future_or_result
            )
        except Exception as e:
            # Always surfaced: an error is not routine per-frame chatter.
            safe_print(f"Error processing frame {count}: {e}")
            tracker.record(count, False)  # Errored frame counts as done (skipped)
            save_progress()
            return
        reporter.log(result)
        tracker.record(count, saved)
        all_filters_failed.update(filters_failed)
        save_progress()

    reporter = _StatusReporter(expected_total, start_at, verbose)
    try:
        if use_parallel:
            max_in_flight = (os.cpu_count() or 4) * 2
            with ThreadPoolExecutor() as executor:
                in_flight = set()
                future_count = {}  # future -> count, so we know the index even on error
                for frame, count in frame_generator():
                    if count < start_at:
                        continue
                    fut = executor.submit(process_frame, make_task_args(frame, count))
                    future_count[fut] = count
                    in_flight.add(fut)
                    if len(in_flight) >= max_in_flight:
                        done, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
                        for future in done:
                            handle_result(future_count.pop(future), future)
                            reporter.advance(tracker)
                for future in as_completed(in_flight):
                    handle_result(future_count.pop(future), future)
                    reporter.advance(tracker)
        else:
            for frame, count in frame_generator():
                if count < start_at:
                    continue
                handle_result(count, process_frame(make_task_args(frame, count)))
                reporter.advance(tracker)
    finally:
        reporter.close(tracker)
        # Release the capture on the error path too, or an interrupted run
        # keeps the video file open until the interpreter exits - which on
        # Windows blocks the caller from deleting the downloaded file.
        video.release()

    # Extraction finished successfully - remove the progress file so a future
    # --resume run doesn't skip frames of a new extraction.
    if resume and os.path.exists(progress_file):
        os.unlink(progress_file)

    return tracker.processed, tracker.skipped, tracker.saved, all_filters_failed

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

    # Saved frames do NOT all share one size: remove_black_bars() crops each
    # frame to its own content, so letterboxed and pillarboxed frames come out
    # smaller than the rest. Fit every tile into a common cell instead of
    # assuming the first frame's size, which otherwise leaves the grid ragged
    # with black gaps and overlapping pastes.
    cols = min(3, len(selected_frames))
    rows = (len(selected_frames) + cols - 1) // cols

    # Open each selected frame once: the size pass and the paste pass used to
    # read every image from disk twice.
    tiles = []
    for f in selected_frames:
        with Image.open(os.path.join(output_folder, f)) as im:
            tiles.append(im.convert('RGB'))
    sizes = [t.size for t in tiles]
    cell_w = max(w for w, _ in sizes)
    cell_h = max(h for _, h in sizes)

    thumbnail = Image.new('RGB', (cell_w * cols, cell_h * rows), (0, 0, 0))

    for i, tile in enumerate(tiles):
        # Preserve aspect ratio, then centre the tile inside its cell.
        tile.thumbnail((cell_w, cell_h), Image.LANCZOS)
        x = (i % cols) * cell_w + (cell_w - tile.width) // 2
        y = (i // cols) * cell_h + (cell_h - tile.height) // 2
        thumbnail.paste(tile, (x, y))

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
    XX: Quality score (0-100, higher is better)
    YY: Blur score (higher numbers indicate less blur)
  Both scores are measured on the frame as cropped, before any optional
  --gradfun/--deblock/--deband filtering (deblock is a denoiser, so it lowers
  the blur score by design).
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
- Per-frame lines are printed only with --verbose. Without it you get a progress
  bar in a terminal, or a periodic one-line summary when output is piped.
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
    parser.add_argument("--quality", type=float, default=30.0,
                        help="Quality threshold for frame selection (0-100, higher is stricter, default: 30.0)")
    parser.add_argument("--blur", type=float, default=50.0,
                        help="Blur threshold for frame selection (higher allows less blur, default: 50.0)")
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
                        help=argparse.SUPPRESS)  # Deprecated: kept for backwards compatibility, has no effect
    parser.add_argument("--fast-scene", action="store_true",
                        help="Use fast mode for scene detection (less accurate results)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume an interrupted extraction process")
    parser.add_argument("--thumbnail", action="store_true",
                        help="Generate a thumbnail montage of extracted frames")
    parser.add_argument("--keep-video", action="store_true",
                        help="Keep the downloaded source video instead of deleting it after extraction (ignored for local files)")
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
    parser.add_argument("--extractor-args", type=str, metavar='ARGS', action='append',
                        help="Additional extractor arguments for yt-dlp in EXTRACTOR:ARG=VALUE form "
                             "(e.g., 'youtube:player_client=mweb'). May be given multiple times. See yt-dlp documentation.")

    args = parser.parse_args()

    if args.config:
        # Re-parse so explicit command-line flags still win over the file.
        config_appends = apply_config_file(parser, args.config)
        args = parser.parse_args()
        for dest, value in config_appends.items():
            if getattr(args, dest) is None:  # nothing given on the command line
                setattr(args, dest, value)

    if args.quality < 0 or args.quality > 100:
        parser.error("Quality threshold must be between 0 and 100.")

    if args.blur < 0 or args.blur > 1000:
        parser.error("Blur threshold must be between 0 and 1000.")

    if args.watermark_threshold < 0 or args.watermark_threshold > 1:
        parser.error("Watermark threshold must be between 0 and 1.")

    if args.interval <= 0:
        parser.error("Interval must be greater than 0.")

    if args.cookies and not os.path.isfile(args.cookies):
        parser.error(f"Cookies file not found: {args.cookies}")

    if args.method == 'keyframes':
        if not check_ffmpeg():
            print("Error: The 'keyframes' method requires FFmpeg.")
            print("Please install FFmpeg from https://ffmpeg.org/download.html")
            print("Or use the startup script option 4 to install it.")
            sys.exit(1)

    if args.use_gpu:
        print("Note: --use-gpu is deprecated and has no effect. CPU processing is used.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parse extractor_args into yt-dlp's nested {ie: {arg: [values]}} format
    extractor_args_dict = None
    if args.extractor_args:
        try:
            extractor_args_dict = parse_extractor_args(args.extractor_args)
        except ValueError as e:
            parser.error(str(e))

    is_url = args.source.startswith(('http://', 'https://', 'www.'))

    if is_url:
        video_path = f"downloaded_video_{timestamp}.mp4"

        # Clean URL - remove playlist params that cause wrong video extraction
        cleaned_url = clean_youtube_url(args.source)
        if cleaned_url != args.source:
            safe_print("Note: Stripped playlist parameters from URL")
            safe_print(f"  Original: {args.source}")
            safe_print(f"  Cleaned:  {cleaned_url}")

        if args.dry_run:
            # Fetch metadata only - do not download the video during a dry run
            safe_print("Dry run: fetching video info (no download)...")
            info = get_video_info(
                cleaned_url, max_resolution=args.max_resolution, verbose=args.verbose,
                cookies_from_browser=args.cookies_from_browser, cookies_file=args.cookies,
                sleep_requests=args.sleep_requests, extractor_args=extractor_args_dict
            )
            video_title = info.get('title', 'Unknown')
            duration = info.get('duration')
            safe_print(f"Video found: {video_title}")
            if duration:
                safe_print(f"Duration: {int(duration // 60)}m {int(duration % 60)}s")
        else:
            try:
                video_title = download_video(
                    cleaned_url, video_path, args.max_resolution, args.verbose,
                    cookies_from_browser=args.cookies_from_browser,
                    cookies_file=args.cookies,
                    sleep_requests=args.sleep_requests,
                    extractor_args=extractor_args_dict
                )
            except BaseException:
                # A failed download leaves the half-written target plus
                # yt-dlp's .part bookkeeping behind; clean them up instead of
                # accumulating multi-hundred-MB orphans across failed runs.
                remove_partial_download(video_path)
                raise
        sanitized_title = sanitize_filename(video_title)
    else:
        # It's a local file
        video_path = args.source
        if not os.path.isfile(video_path):
            print(f"Error: Video file not found: {video_path}")
            sys.exit(1)
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
        print(f"Extractor arguments: {'; '.join(args.extractor_args)}")

    if not args.dry_run:
        start_time = time.time()
        try:
            total_frames, skipped_frames, saved_frames, filters_failed = extract_frames(
                video_path, output_folder, args.method, args.interval, args.quality,
                args.blur, args.detect_watermarks, args.watermark_threshold,
                not args.disable_parallel, args.png, args.fast_scene,
                args.resume, args.verbose, args.gradfun, args.deblock, args.deband
            )
        except BaseException:
            # Same reasoning as the download path above: an aborted extraction
            # should not leave our downloaded video behind either.
            if is_url and not args.keep_video:
                remove_partial_download(video_path)
            raise
        end_time = time.time()

        execution_time = end_time - start_time
        frames_per_second = total_frames / execution_time if execution_time > 0 else 0

        print("\nFrame extraction complete.")
        print(f"Total execution time: {execution_time:.2f} seconds")
        print(f"Processed {total_frames} frames.")
        print(f"{saved_frames} high-quality frame{'s' if saved_frames != 1 else ''} saved!")
        print(f"{skipped_frames} frame{'s' if skipped_frames != 1 else ''} skipped due to low-quality and/or blur.")
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

        # Remove the downloaded source video (URL inputs only) so runs don't
        # leave full-size files behind. Never touch a user's local file.
        if is_url and not args.keep_video:
            try:
                if os.path.exists(video_path):
                    os.remove(video_path)
                    safe_print(f"Cleaned up downloaded video: {video_path}")
            except OSError as e:
                safe_print(f"Note: Could not remove downloaded video {video_path}: {e}")
        elif is_url and args.keep_video:
            safe_print(f"Downloaded video kept at: {video_path}")
    else:
        print("Dry run completed. No video was downloaded and no frames were processed.")

if __name__ == "__main__":
    main()
