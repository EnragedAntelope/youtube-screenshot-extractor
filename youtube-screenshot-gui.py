#!/usr/bin/env python3
"""
YouTube Screenshot Extractor - GUI
A graphical interface for extracting frames from videos.
"""

from collections import deque
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import sys
import os
import threading
import time


def check_ffmpeg():
    """Check if FFmpeg is available in the system PATH."""
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                      creationflags=creationflags)
        return True
    except OSError:
        return False


def _kill_process_tree(process):
    """Stop an extraction and everything it spawned.

    terminate() kills only the direct child (the CLI); yt-dlp's FFmpeg
    survives on Windows as an orphan and keeps running headless. taskkill /T
    walks the whole process tree instead. Nothing graceful is lost: the CLI
    has no signal handler, so a plain terminate skips its cleanup anyway.
    """
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       creationflags=subprocess.CREATE_NO_WINDOW)
    else:
        process.terminate()


def wheel_steps(event):
    """Normalise a wheel event into scroll units (positive = scroll down).

    Windows reports multiples of 120 in event.delta, macOS reports small
    counts, and X11 sends Button-4 (up) / Button-5 (down) with no delta.
    """
    if getattr(event, "num", None) == 4:
        return -1
    if getattr(event, "num", None) == 5:
        return 1
    delta = getattr(event, "delta", 0)
    if not delta:
        return 0
    if abs(delta) >= 120:
        return int(-delta / 120)
    return -1 if delta > 0 else 1


class ToolTip:
    """Tooltip that appears on hover."""

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, event=None):
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5

        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            self.tooltip,
            text=self.text,
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9),
            wraplength=300,
            justify="left",
            padx=6,
            pady=3
        )
        label.pack()

    def hide(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None


class ScrollableFrame(ttk.Frame):
    """A frame that supports scrolling when content exceeds window size."""

    def __init__(self, container, *args, **kwargs):
        super().__init__(container, *args, **kwargs)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: self._update_scroll_region())
        self.canvas_frame = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

    def _update_scroll_region(self):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self._check_scrollbar()

    def _bind_mousewheel(self, event):
        # X11 (Linux) reports wheel motion as Button-4/Button-5 rather than
        # <MouseWheel>, so bind both families or scrolling is dead there.
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if not self.scrollbar.winfo_ismapped():
            return
        steps = wheel_steps(event)
        if steps == 0:
            return
        current = self.canvas.yview()
        if steps < 0 and current[0] <= 0:
            return
        if steps > 0 and current[1] >= 1:
            return
        self.canvas.yview_scroll(steps, "units")

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_frame, width=event.width)
        self._check_scrollbar()

    def _check_scrollbar(self):
        self.update_idletasks()
        if self.scrollable_frame.winfo_reqheight() > self.canvas.winfo_height():
            self.scrollbar.pack(side="right", fill="y")
        else:
            self.scrollbar.pack_forget()
            self.canvas.yview_moveto(0)


class YouTubeScreenshotGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube Screenshot Extractor")
        self.root.geometry("660x660")
        self.root.minsize(480, 400)

        # Check FFmpeg availability at startup
        self.ffmpeg_available = check_ffmpeg()

        # Track output window for reuse
        self.output_window = None
        self.output_text = None
        # Every output line ever produced, so a closed log window can be
        # reopened with its history intact (the Text widget dies with it).
        self.output_buffer = deque(maxlen=20000)
        # Currently running extraction subprocess, if any
        self.process = None
        self.stop_requested = False
        # Throttle for @@PROGRESS-driven bar updates (set by the reader thread).
        self._last_progress_ui = 0.0

        try:
            self.root.iconbitmap("icon.ico")
        except tk.TclError:
            # No icon file, or the platform's Tk does not support .ico files.
            pass

        # Configure styles - compact fonts
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("Segoe UI", 9))
        self.style.configure("TButton", font=("Segoe UI", 9))
        self.style.configure("TCheckbutton", font=("Segoe UI", 9))
        self.style.configure("TRadiobutton", font=("Segoe UI", 9))
        self.style.configure("Header.TLabel", font=("Segoe UI", 9, "bold"))
        self.style.configure("Run.TButton", font=("Segoe UI", 10, "bold"))
        self.style.configure("Note.TLabel", font=("Segoe UI", 8, "italic"), foreground="#555555")

        # Create scrollable container
        self.scroll_container = ScrollableFrame(root)
        self.scroll_container.pack(fill="both", expand=True, padx=8, pady=4)
        self.main_frame = self.scroll_container.scrollable_frame

        self._init_variables()
        self._create_input_section()
        self._create_output_section()
        self._create_method_section()
        self._create_quality_section()
        self._create_options_section()
        self._create_youtube_auth_section()
        self._create_action_section()
        self._create_status_bar()
        # Match the controls to the starting method. Only the radio callback
        # would otherwise do this, so changing the default method to keyframes
        # would open the window with its ignored options still live.
        self._sync_pipeline_controls(self.method_var.get())

    def _init_variables(self):
        self.source_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.method_var = tk.StringVar(value="scene")  # Scene detection finds natural cuts
        self.interval_var = tk.DoubleVar(value=5.0)
        self.quality_var = tk.DoubleVar(value=30.0)
        self.blur_var = tk.DoubleVar(value=50.0)
        self.max_resolution_var = tk.StringVar(value="1080")  # Good balance of quality vs speed/rate limits
        self.parallel_var = tk.BooleanVar(value=True)
        self.detect_watermarks_var = tk.BooleanVar(value=True)
        self.watermark_threshold_var = tk.DoubleVar(value=0.8)
        self.fast_scene_var = tk.BooleanVar(value=False)
        self.resume_var = tk.BooleanVar(value=False)
        self.thumbnail_var = tk.BooleanVar(value=True)
        self.png_var = tk.BooleanVar(value=False)
        self.keep_video_var = tk.BooleanVar(value=False)
        self.verbose_var = tk.BooleanVar(value=False)
        self.gradfun_var = tk.BooleanVar(value=False)
        self.deblock_var = tk.BooleanVar(value=False)  # Off by default: runs per-frame denoising (slow)
        self.deband_var = tk.BooleanVar(value=False)
        # YouTube authentication options
        self.cookies_from_browser_var = tk.StringVar(value="")
        self.cookies_file_var = tk.StringVar(value="")
        self.sleep_requests_var = tk.IntVar(value=0)
        self.extractor_args_var = tk.StringVar(value="")

    def _create_section_header(self, parent, text):
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(8, 2))
        label = ttk.Label(frame, text=text, style="Header.TLabel")
        label.pack(side="left")
        ttk.Separator(frame, orient="horizontal").pack(side="left", fill="x", expand=True, padx=(8, 0))
        return frame

    def _create_input_section(self):
        self._create_section_header(self.main_frame, "Video Source")

        frame = ttk.Frame(self.main_frame)
        frame.pack(fill="x", pady=2)

        self.source_entry = ttk.Entry(frame, textvariable=self.source_var)
        self.source_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ToolTip(self.source_entry, "Paste a YouTube URL or use Browse for local file")

        ttk.Button(frame, text="Browse", command=self._browse_video, width=8).pack(side="right")

        # Resolution row
        res_frame = ttk.Frame(self.main_frame)
        res_frame.pack(fill="x", pady=2)
        ttk.Label(res_frame, text="Max Resolution:").pack(side="left")
        res_combo = ttk.Combobox(res_frame, textvariable=self.max_resolution_var, width=8,
                                  values=["", "480", "720", "1080", "1440", "2160"],
                                  state="readonly")
        res_combo.pack(side="left", padx=(4, 0))
        ToolTip(res_combo, "Limit YouTube download quality. Higher = slower downloads, more rate limiting. 1080 recommended.")

    def _create_output_section(self):
        self._create_section_header(self.main_frame, "Output")

        frame = ttk.Frame(self.main_frame)
        frame.pack(fill="x", pady=2)

        self.output_entry = ttk.Entry(frame, textvariable=self.output_var)
        self.output_entry.pack(side="left", fill="x", expand=True, padx=(0, 4))
        ToolTip(self.output_entry, "Output folder. Leave empty for auto-generated name.")

        ttk.Button(frame, text="Browse", command=self._browse_output, width=8).pack(side="right")

        # Format row
        fmt_frame = ttk.Frame(self.main_frame)
        fmt_frame.pack(fill="x", pady=2)
        ttk.Label(fmt_frame, text="Format:").pack(side="left")
        ttk.Radiobutton(fmt_frame, text="JPG", variable=self.png_var, value=False).pack(side="left", padx=(8, 4))
        ttk.Radiobutton(fmt_frame, text="PNG", variable=self.png_var, value=True).pack(side="left")

    def _create_method_section(self):
        self._create_section_header(self.main_frame, "Extraction Method")

        self.method_radio_frame = ttk.Frame(self.main_frame)
        self.method_radio_frame.pack(fill="x", pady=2)

        # Methods with tooltips explaining performance implications
        method_info = [
            ("Interval", "interval", "Extract frame every N seconds. Can be slow for long videos with short intervals.", True),
            ("Keyframes", "keyframes", "Extract I-frames only. Fast, requires FFmpeg.", self.ffmpeg_available),
            ("Scene", "scene", "Detect scene changes. Best quality, moderate speed.", True),
            ("All", "all", "Extract every frame. VERY slow and generates huge output - use with caution!", True),
        ]
        for text, value, tooltip, enabled in method_info:
            rb = ttk.Radiobutton(self.method_radio_frame, text=text, variable=self.method_var,
                                value=value, command=self._on_method_change)
            rb.pack(side="left", padx=(0, 12))
            if not enabled:
                rb.configure(state="disabled")
                tooltip = tooltip + " (FFmpeg not found - install via startup script option 4)"
            ToolTip(rb, tooltip)

        # Interval setting (hidden by default since scene is default method)
        self.interval_frame = ttk.Frame(self.main_frame)
        ttk.Label(self.interval_frame, text="Interval (sec):").pack(side="left")
        ttk.Spinbox(self.interval_frame, from_=0.1, to=60, increment=0.5,
                   textvariable=self.interval_var, width=6).pack(side="left", padx=(4, 0))

        # Fast scene option (shown by default since scene is default method)
        self.fast_scene_frame = ttk.Frame(self.main_frame)
        self.fast_scene_frame.pack(fill="x", pady=2, after=self.method_radio_frame)
        ttk.Checkbutton(self.fast_scene_frame, text="Fast scene detection (less accurate)",
                       variable=self.fast_scene_var).pack(side="left")

    def _create_quality_section(self):
        self._create_section_header(self.main_frame, "Quality Thresholds")

        # Quality slider
        q_frame = ttk.Frame(self.main_frame)
        q_frame.pack(fill="x", pady=2)
        ttk.Label(q_frame, text="Quality:", width=8).pack(side="left")
        self.quality_value_label = ttk.Label(q_frame, text="30", width=4)
        self.quality_value_label.pack(side="right")
        self.quality_scale = ttk.Scale(q_frame, from_=0, to=100, variable=self.quality_var,
                 command=lambda v: self.quality_value_label.config(text=f"{float(v):.0f}"))
        self.quality_scale.pack(side="left", fill="x", expand=True, padx=4)
        ToolTip(q_frame, "Min quality 0-100. Higher = stricter. Default 30; raise toward 50 to be pickier.")

        # Blur slider
        b_frame = ttk.Frame(self.main_frame)
        b_frame.pack(fill="x", pady=2)
        ttk.Label(b_frame, text="Blur:", width=8).pack(side="left")
        self.blur_value_label = ttk.Label(b_frame, text="50", width=4)
        self.blur_value_label.pack(side="right")
        self.blur_scale = ttk.Scale(b_frame, from_=0, to=1000, variable=self.blur_var,
                 command=lambda v: self.blur_value_label.config(text=f"{float(v):.0f}"))
        self.blur_scale.pack(side="left", fill="x", expand=True, padx=4)
        ToolTip(b_frame, "Min sharpness. Higher = less blur allowed. Default 50; raise toward 150 to be pickier.")

    def _create_options_section(self):
        self._create_section_header(self.main_frame, "Options")

        # Row 1: Performance
        row1 = ttk.Frame(self.main_frame)
        row1.pack(fill="x", pady=2)
        par_cb = ttk.Checkbutton(row1, text="Parallel processing", variable=self.parallel_var)
        par_cb.pack(side="left")
        ToolTip(par_cb, "Process multiple frames simultaneously.")
        self.parallel_cb = par_cb

        # Row 2: Features
        row2 = ttk.Frame(self.main_frame)
        row2.pack(fill="x", pady=2)
        wm_cb = ttk.Checkbutton(row2, text="Detect watermarks", variable=self.detect_watermarks_var)
        wm_cb.pack(side="left", padx=(0, 4))
        ToolTip(wm_cb, "Mark frames with potential watermarks.")

        wm_thresh = ttk.Spinbox(row2, from_=0.0, to=1.0, increment=0.05, width=5,
                                textvariable=self.watermark_threshold_var, format="%.2f")
        wm_thresh.pack(side="left", padx=(0, 16))
        ToolTip(wm_thresh, "Watermark sensitivity 0-1. Higher = fewer false positives. Default 0.8.")

        th_cb = ttk.Checkbutton(row2, text="Thumbnail", variable=self.thumbnail_var)
        th_cb.pack(side="left", padx=(0, 16))
        ToolTip(th_cb, "Generate 3x3 preview montage.")

        res_cb = ttk.Checkbutton(row2, text="Resume", variable=self.resume_var)
        res_cb.pack(side="left", padx=(0, 16))
        ToolTip(res_cb, "Continue from previous extraction.")
        self.watermark_cb, self.watermark_spin, self.resume_cb = wm_cb, wm_thresh, res_cb

        kv_cb = ttk.Checkbutton(row2, text="Keep video", variable=self.keep_video_var)
        kv_cb.pack(side="left")
        ToolTip(kv_cb, "Keep the downloaded source video instead of deleting it after extraction. Ignored for local files.")

        # Row 3: Filters
        row3 = ttk.Frame(self.main_frame)
        row3.pack(fill="x", pady=2)
        ttk.Label(row3, text="Filters:").pack(side="left", padx=(0, 8))

        gf_cb = ttk.Checkbutton(row3, text="Gradfun", variable=self.gradfun_var)
        gf_cb.pack(side="left", padx=(0, 8))
        if self.ffmpeg_available:
            ToolTip(gf_cb, "Subtle color banding reduction. Requires FFmpeg.")
        else:
            gf_cb.configure(state="disabled")
            self.gradfun_var.set(False)
            ToolTip(gf_cb, "Subtle color banding reduction. (FFmpeg not found - install via startup script option 4)")

        db_cb = ttk.Checkbutton(row3, text="Deblock", variable=self.deblock_var)
        db_cb.pack(side="left", padx=(0, 8))
        ToolTip(db_cb, "Reduce compression artifacts. Works without FFmpeg.")

        dband_cb = ttk.Checkbutton(row3, text="Deband", variable=self.deband_var)
        dband_cb.pack(side="left")
        if self.ffmpeg_available:
            ToolTip(dband_cb, "Aggressive color banding reduction. Requires FFmpeg.")
        else:
            dband_cb.configure(state="disabled")
            self.deband_var.set(False)
            ToolTip(dband_cb, "Aggressive color banding reduction. (FFmpeg not found - install via startup script option 4)")

        # Controls that only affect the per-frame pipeline. The keyframes method
        # shells straight to FFmpeg and never runs that pipeline, so these are
        # greyed out while it is selected rather than silently ignored.
        # gradfun/deband are listed as FFmpeg-gated: when FFmpeg is missing they
        # are permanently disabled and must not be re-enabled by a method change.
        self._pipeline_widgets = [
            (self.quality_scale, False), (self.blur_scale, False),
            (self.parallel_cb, False), (self.watermark_cb, False),
            (self.watermark_spin, False), (self.resume_cb, False),
            (gf_cb, True), (db_cb, False), (dband_cb, True),
        ]

        # Explains the greying out; packed only while keyframes is selected.
        self.keyframes_note = ttk.Label(
            self.main_frame, style="Note.TLabel", wraplength=600, justify="left",
            text=("Keyframes mode extracts I-frames directly with FFmpeg, so the greyed-out "
                  "options above (quality/blur thresholds, watermark detection, filters, "
                  "resume, parallel) do not apply. Format and Thumbnail still do."),
        )
        # Wrap to the width actually available: a fixed wraplength wider than
        # the window clips the tail of the note, and the panel has no
        # horizontal scrollbar to reveal it. Only reassign on a real change,
        # or setting it from inside <Configure> re-triggers this handler.
        def _rewrap(event):
            target = max(200, event.width - 8)
            if self.keyframes_note.cget("wraplength") != target:
                self.keyframes_note.configure(wraplength=target)
        self.keyframes_note.bind("<Configure>", _rewrap)

    def _create_youtube_auth_section(self):
        """Create YouTube authentication and rate limiting section."""
        self._create_section_header(self.main_frame, "YouTube Authentication (Optional)")

        # Cookies from browser row
        browser_frame = ttk.Frame(self.main_frame)
        browser_frame.pack(fill="x", pady=2)
        ttk.Label(browser_frame, text="Browser Cookies:").pack(side="left")
        browser_combo = ttk.Combobox(browser_frame, textvariable=self.cookies_from_browser_var,
                                     values=["", "firefox", "chrome", "edge", "safari"],
                                     width=10, state="readonly")
        browser_combo.pack(side="left", padx=(4, 0))
        ToolTip(browser_combo, "Select browser to use cookies from. Required for age-restricted videos and helps with PO Token issues. Leave empty if not needed.")

        # Cookies file row (alternative to browser cookies)
        cfile_frame = ttk.Frame(self.main_frame)
        cfile_frame.pack(fill="x", pady=2)
        ttk.Label(cfile_frame, text="Cookies File:").pack(side="left")
        cfile_entry = ttk.Entry(cfile_frame, textvariable=self.cookies_file_var)
        cfile_entry.pack(side="left", fill="x", expand=True, padx=(4, 4))
        ToolTip(cfile_entry, "Path to a Netscape-format cookies file. Alternative to browser cookies - ignored if a browser is selected above.")
        ttk.Button(cfile_frame, text="Browse", command=self._browse_cookies_file, width=8).pack(side="right")

        # Rate limiting row
        rate_frame = ttk.Frame(self.main_frame)
        rate_frame.pack(fill="x", pady=2)
        ttk.Label(rate_frame, text="Rate Limit (s):").pack(side="left")
        ttk.Spinbox(rate_frame, from_=0, to=60, increment=1,
                    textvariable=self.sleep_requests_var, width=6).pack(side="left", padx=(4, 0))
        ToolTip(rate_frame, "Delay between requests (seconds). Use 3-5 when processing multiple videos to avoid rate limiting. 0 = no delay.")

        # Extractor args row (advanced)
        ea_frame = ttk.Frame(self.main_frame)
        ea_frame.pack(fill="x", pady=2)
        ttk.Label(ea_frame, text="Extractor Args:").pack(side="left")
        ea_entry = ttk.Entry(ea_frame, textvariable=self.extractor_args_var)
        ea_entry.pack(side="left", fill="x", expand=True, padx=(4, 0))
        ToolTip(ea_entry, "Advanced: extra yt-dlp extractor args, e.g. 'youtube:player_client=mweb'. Leave empty if unsure.")

    def _create_action_section(self):
        frame = ttk.Frame(self.main_frame)
        frame.pack(fill="x", pady=(12, 4))

        verbose_cb = ttk.Checkbutton(frame, text="Verbose", variable=self.verbose_var)
        verbose_cb.pack(side="left")
        ToolTip(verbose_cb, "Show detailed yt-dlp output (noisy).")

        self.log_button = ttk.Button(frame, text="Show Log", command=self._show_output_log,
                                     width=10)
        self.log_button.pack(side="left", padx=(8, 0))
        ToolTip(self.log_button, "Reopen the extraction output log.")

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(side="right")

        self.stop_button = ttk.Button(btn_frame, text="Stop", command=self._stop, width=8,
                                      state="disabled")
        self.stop_button.pack(side="left", padx=(0, 8))
        ToolTip(self.stop_button, "Stop the running extraction. Frames already saved are kept.")

        self.dry_run_button = ttk.Button(btn_frame, text="Dry Run", command=self._dry_run, width=10)
        self.dry_run_button.pack(side="left", padx=(0, 8))

        self.run_button = ttk.Button(btn_frame, text="Extract", style="Run.TButton",
                                     command=self._run, width=10)
        self.run_button.pack(side="left")

    def _create_status_bar(self):
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken",
                               anchor="w", font=("Segoe UI", 9), padding=(4, 2))
        status_bar.pack(side="bottom", fill="x")
        # Indeterminate until the CLI reports frame counts; hidden when idle.
        self.progress_bar = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress_bar.pack(side="bottom", fill="x", padx=8, pady=(3, 0))
        self.progress_bar.pack_forget()

    def _on_method_change(self):
        method = self.method_var.get()
        if method == "interval":
            self.fast_scene_frame.pack_forget()
            self.interval_frame.pack(fill="x", pady=2, after=self.method_radio_frame)
        elif method == "scene":
            self.interval_frame.pack_forget()
            self.fast_scene_frame.pack(fill="x", pady=2, after=self.method_radio_frame)
        else:
            self.interval_frame.pack_forget()
            self.fast_scene_frame.pack_forget()
        self._sync_pipeline_controls(method)

    def _sync_pipeline_controls(self, method):
        """Grey out the options the chosen method ignores.

        keyframes shells straight to FFmpeg and never runs the per-frame
        pipeline, so leaving these live invites the exact failure this repo
        keeps hitting: a setting the user changed that quietly does nothing.
        """
        keyframes = method == "keyframes"
        for widget, needs_ffmpeg in self._pipeline_widgets:
            # An FFmpeg-gated control stays disabled when FFmpeg is missing;
            # re-enabling it here would resurrect an option that cannot work.
            if needs_ffmpeg and not self.ffmpeg_available:
                continue
            widget.configure(state="disabled" if keyframes else "normal")
        if keyframes:
            self.keyframes_note.pack(fill="x", pady=(2, 4), after=self.method_radio_frame)
        else:
            self.keyframes_note.pack_forget()

    def _browse_video(self):
        filetypes = [("Video files", "*.mp4 *.mkv *.avi *.mov *.webm *.flv *.wmv"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(title="Select Video", filetypes=filetypes)
        if filename:
            self.source_var.set(filename)

    def _browse_output(self):
        folder = filedialog.askdirectory(title="Select Output Folder")
        if folder:
            self.output_var.set(folder)

    def _browse_cookies_file(self):
        filetypes = [("Cookies files", "*.txt"), ("All files", "*.*")]
        filename = filedialog.askopenfilename(title="Select Cookies File", filetypes=filetypes)
        if filename:
            self.cookies_file_var.set(filename)

    def _read_number(self, var, label, low, high):
        """Read a numeric entry, reporting bad input instead of crashing.

        The Spinboxes are freely typeable, and a DoubleVar/IntVar whose entry
        holds something non-numeric raises TclError on get(). Uncaught, that
        propagates out of the button callback into Tk's error handler - which
        under pythonw has nowhere to print - so the button just looked dead.
        Returns None after telling the user which field is wrong.
        """
        try:
            value = var.get()
        except tk.TclError:
            messagebox.showerror("Invalid value", f"{label} must be a number.")
            return None
        if not low <= value <= high:
            messagebox.showerror("Invalid value", f"{label} must be between {low} and {high}.")
            return None
        return value

    def _build_command(self, dry_run=False):
        source = self.source_var.get().strip()
        if not source:
            messagebox.showerror("Error", "Please enter a video URL or select a local file.")
            return None

        cmd = [sys.executable, "youtube-screenshot-script.py", source]
        method = self.method_var.get()
        cmd.extend(["--method", method])

        if method == "interval":
            # 0.1 is the Spinbox floor, but the entry is typeable, and the CLI
            # rejects an interval of 0 - catch it here with a usable message.
            interval = self._read_number(self.interval_var, "Interval", 0.1, 3600)
            if interval is None:
                return None
            cmd.extend(["--interval", str(interval)])
        if method == "scene" and self.fast_scene_var.get():
            cmd.append("--fast-scene")

        # keyframes never runs the per-frame pipeline, so the options that only
        # feed it are left off the command line entirely - matching the controls
        # _sync_pipeline_controls() greys out. Passing them would be accepted and
        # ignored, which is how they came to look functional in the first place.
        pipeline = method != "keyframes"

        if pipeline:
            cmd.extend(["--quality", f"{self.quality_var.get():.0f}"])
            cmd.extend(["--blur", f"{self.blur_var.get():.0f}"])

        output = self.output_var.get().strip()
        if output:
            cmd.extend(["--output", output])

        max_res = self.max_resolution_var.get().strip()
        if max_res:
            cmd.extend(["--max-resolution", max_res])

        if self.png_var.get():
            cmd.append("--png")
        if pipeline and not self.parallel_var.get():
            cmd.append("--disable-parallel")
        if pipeline and self.detect_watermarks_var.get():
            threshold = self._read_number(
                self.watermark_threshold_var, "Watermark threshold", 0.0, 1.0)
            if threshold is None:
                return None
            cmd.append("--detect-watermarks")
            cmd.extend(["--watermark-threshold", str(threshold)])
        if self.thumbnail_var.get():
            cmd.append("--thumbnail")
        if pipeline and self.resume_var.get():
            cmd.append("--resume")
        if self.keep_video_var.get():
            cmd.append("--keep-video")
        if pipeline and self.gradfun_var.get():
            cmd.append("--gradfun")
        if pipeline and self.deblock_var.get():
            cmd.append("--deblock")
        if pipeline and self.deband_var.get():
            cmd.append("--deband")
        if self.verbose_var.get():
            cmd.append("--verbose")
        if dry_run:
            cmd.append("--dry-run")

        # Add YouTube authentication options
        cookies_browser = self.cookies_from_browser_var.get()
        cookies_file = self.cookies_file_var.get().strip()
        if cookies_browser:
            cmd.extend(["--cookies-from-browser", cookies_browser])
        elif cookies_file:
            cmd.extend(["--cookies", cookies_file])

        sleep_requests = self._read_number(self.sleep_requests_var, "Rate limit", 0, 3600)
        if sleep_requests is None:
            return None
        if sleep_requests > 0:
            cmd.extend(["--sleep-requests", str(int(sleep_requests))])

        extractor_args = self.extractor_args_var.get().strip()
        if extractor_args:
            cmd.extend(["--extractor-args", extractor_args])

        return cmd

    def _run_command(self, cmd, dry_run=False):
        # One extraction at a time: two runs started by an impatient double-click
        # would write into the same output folder and interleave their output.
        if self.process is not None and self.process.poll() is None:
            messagebox.showinfo("Already running",
                                "An extraction is already in progress. Wait for it to "
                                "finish, or press Stop.")
            return

        self.stop_requested = False
        self._set_running(True)
        self._start_progress_bar()
        self.status_var.set("Dry run..." if dry_run else "Processing...")
        self._create_output_window(dry_run, cmd)

        def run():
            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))

                # Create process with unbuffered output
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                # Ask the CLI for @@PROGRESS machine lines (parsed and stripped
                # by the reader thread) to drive the determinate progress bar.
                env['YSE_PROGRESS'] = '1'

                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=script_dir, env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                    bufsize=1  # Line buffered
                )
                self.process = process
                # Honour a Stop pressed between _set_running(True) and the
                # assignment above, when there was no process to kill yet.
                if self.stop_requested:
                    try:
                        _kill_process_tree(process)
                    except OSError:
                        pass
                self._read_output(process)
            except Exception as e:
                # Tk is not thread-safe, so every widget touch from this worker
                # thread has to be marshalled back onto the main loop.
                msg = str(e)
                self.root.after(0, lambda: self._on_process_finished(None, msg))

        threading.Thread(target=run, daemon=True).start()

    def _set_running(self, running):
        state = "disabled" if running else "normal"
        self.run_button.configure(state=state)
        self.dry_run_button.configure(state=state)
        self.stop_button.configure(state="normal" if running else "disabled")

    def _stop(self):
        # Recorded before the process lookup: the worker thread assigns
        # self.process only once Popen has returned, so a Stop pressed in that
        # window finds nothing to kill. The flag makes the intent stick, and
        # the worker honours it as soon as the child exists.
        self.stop_requested = True
        self.status_var.set("Stopping...")
        process = self.process
        if process is None or process.poll() is not None:
            return
        try:
            _kill_process_tree(process)
        except OSError as e:
            messagebox.showerror("Error", f"Could not stop the extraction: {e}")

    def _on_process_finished(self, returncode, error=None):
        self.process = None
        self._set_running(False)
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        if error is not None:
            self.status_var.set("Error")
            self._append_output(self.output_text, f"\nError: {error}\n")
            messagebox.showerror("Error", error)
            return
        if self.stop_requested:
            # A user stop surfaces as whatever code the platform kill produces
            # (-SIGTERM on POSIX, taskkill's exit code on Windows); report what
            # the user actually did rather than the raw number.
            status = "Stopped. Frames already saved were kept."
        elif returncode == 0:
            status = "Complete!"
        else:
            status = f"Exit code: {returncode}"
        self.stop_requested = False
        self.status_var.set(status)
        self._append_output(self.output_text, f"\n--- {status} ---\n")

    def _build_output_window(self):
        """Create a fresh output log window and replay the buffered history."""
        self.output_window = tk.Toplevel(self.root)
        self.output_window.title("Output")
        self.output_window.geometry("800x500")
        self.output_window.transient(self.root)

        text_frame = ttk.Frame(self.output_window)
        text_frame.pack(fill="both", expand=True, padx=8, pady=8)

        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        self.output_text = tk.Text(text_frame, wrap="word", font=("Consolas", 9), yscrollcommand=scrollbar.set)
        self.output_text.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=self.output_text.yview)

        def on_wheel(ev):
            self.output_text.yview_scroll(wheel_steps(ev), "units")

        def bind_scroll(e):
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.output_text.bind_all(seq, on_wheel)

        def unbind_scroll(e):
            for seq in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                self.output_text.unbind_all(seq)
        self.output_text.bind("<Enter>", bind_scroll)
        self.output_text.bind("<Leave>", unbind_scroll)

        def on_close():
            unbind_scroll(None)
            self.output_window.destroy()
            self.output_window = None
            self.output_text = None
        self.output_window.protocol("WM_DELETE_WINDOW", on_close)

        ttk.Button(self.output_window, text="Close", command=on_close).pack(pady=6)

        # Replay everything logged so far - including earlier runs whose
        # window was closed - so reopening loses nothing.
        for line in self.output_buffer:
            self.output_text.insert("end", line)
        self.output_text.see("end")

    def _create_output_window(self, dry_run=False, cmd=None):
        """Create or reuse the output window."""
        if self.output_window is not None and self.output_window.winfo_exists():
            # Add separator for new run
            self._append_output(self.output_text, "\n" + "="*60 + "\n")
            self._append_output(self.output_text, f"{'Dry Run' if dry_run else 'New Extraction'} Started\n")
            self._append_output(self.output_text, "="*60 + "\n\n")
            self.output_window.lift()  # Bring to front
        else:
            self._build_output_window()
        if cmd:
            self._append_output(self.output_text, f"Command: {' '.join(cmd[:5])}...\n")
            self._append_output(self.output_text, f"Full command: {' '.join(cmd)}\n\n")
        self._append_output(self.output_text, "Processing... please wait.\n\n")

    def _show_output_log(self):
        """Reopen the log window after it has been closed mid-run."""
        if self.output_window is not None and self.output_window.winfo_exists():
            self.output_window.lift()
        else:
            self._build_output_window()

    def _read_output(self, process):
        """Pump the child's output into the log. Runs on the worker thread."""
        try:
            for line in iter(process.stdout.readline, ''):
                if not line:
                    break
                if line.startswith("@@PROGRESS"):
                    self._ingest_progress(line)
                    continue
                self.root.after(0, lambda l=line: self._append_output(self.output_text, l))
            process.wait()
        except Exception as e:
            msg = str(e)
            self.root.after(0, lambda: self._on_process_finished(None, msg))
            return
        finally:
            if process.stdout:
                process.stdout.close()
        self.root.after(0, lambda: self._on_process_finished(process.returncode))

    def _start_progress_bar(self):
        """Indeterminate spinner until the CLI reports a frame total."""
        self._last_progress_ui = 0.0
        self.progress_bar.configure(mode="indeterminate", maximum=100, value=0)
        self.progress_bar.pack(side="bottom", fill="x", padx=8, pady=(3, 0))
        self.progress_bar.start(12)

    def _ingest_progress(self, line):
        """Parse one @@PROGRESS line and refresh the bar, throttled.

        Runs on the worker thread: it only computes and schedules, never
        touches widgets directly.
        """
        fields = {}
        for token in line.split()[1:]:
            key, sep, value = token.partition("=")
            if sep:
                fields[key] = value
        try:
            done = int(fields["done"])
            saved = int(fields.get("saved", "0"))
            skipped = int(fields.get("skipped", "0"))
            total = int(fields["total"]) if "total" in fields else None
        except (KeyError, ValueError):
            return  # malformed machine line - ignore rather than log it
        now = time.monotonic()
        if now - self._last_progress_ui < 0.15:
            return
        self._last_progress_ui = now
        self.root.after(0, lambda: self._update_progress(done, total, saved, skipped))

    def _update_progress(self, done, total, saved, skipped):
        """Determinate bar once a frame total is known; counts in the status."""
        if total:
            # stop() halts the indeterminate animation but also resets the
            # value to 0, so it must run before the value is set - and only
            # on the transition, or every update would restart from zero.
            if str(self.progress_bar.cget("mode")) != "determinate":
                self.progress_bar.stop()
                self.progress_bar.configure(mode="determinate")
            self.progress_bar.configure(maximum=total, value=min(done, total))
            counts = f"{done}/{total} frames"
        else:
            counts = f"{done} frames"
        self.status_var.set(f"Processing... {counts} ({saved} saved, {skipped} skipped)")

    def _append_output(self, widget, line):
        # Buffer first: the Text widget may already be closed, and the buffer
        # is what lets a reopened log show the complete history.
        self.output_buffer.append(line)
        try:
            if widget and widget.winfo_exists():
                widget.insert("end", line)
                widget.see("end")
        except tk.TclError:
            pass

    def _run(self):
        cmd = self._build_command()
        if cmd:
            self._run_command(cmd)

    def _dry_run(self):
        cmd = self._build_command(dry_run=True)
        if cmd:
            self._run_command(cmd, dry_run=True)


def main():
    root = tk.Tk()
    app = YouTubeScreenshotGUI(root)

    def on_quit():
        # Kill the child's whole tree so closing the window does not leave an
        # extraction (or its FFmpeg) running headless with nothing reading
        # its output.
        if app.process is not None and app.process.poll() is None:
            try:
                _kill_process_tree(app.process)
            except OSError:
                pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_quit)
    root.mainloop()


if __name__ == "__main__":
    main()
