#!/usr/bin/env python3
"""
YouTube Screenshot Extractor - GUI
A graphical interface for extracting frames from videos.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import sys
import os
import threading


def check_ffmpeg():
    """Check if FFmpeg is available in the system PATH."""
    try:
        creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                      creationflags=creationflags)
        return True
    except FileNotFoundError:
        return False


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
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        if not self.scrollbar.winfo_ismapped():
            return
        current = self.canvas.yview()
        if event.delta > 0 and current[0] <= 0:
            return
        if event.delta < 0 and current[1] >= 1:
            return
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

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
        self.root.geometry("580x620")
        self.root.minsize(480, 400)

        # Check FFmpeg availability at startup
        self.ffmpeg_available = check_ffmpeg()

        # Track output window for reuse
        self.output_window = None
        self.output_text = None

        try:
            self.root.iconbitmap("icon.ico")
        except:
            pass

        # Configure styles - compact fonts
        self.style = ttk.Style()
        self.style.configure("TLabel", font=("Segoe UI", 9))
        self.style.configure("TButton", font=("Segoe UI", 9))
        self.style.configure("TCheckbutton", font=("Segoe UI", 9))
        self.style.configure("TRadiobutton", font=("Segoe UI", 9))
        self.style.configure("Header.TLabel", font=("Segoe UI", 9, "bold"))
        self.style.configure("Run.TButton", font=("Segoe UI", 10, "bold"))

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

    def _init_variables(self):
        self.source_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.method_var = tk.StringVar(value="scene")  # Scene detection finds natural cuts
        self.interval_var = tk.DoubleVar(value=5.0)
        self.quality_var = tk.DoubleVar(value=50.0)
        self.blur_var = tk.DoubleVar(value=100.0)
        self.max_resolution_var = tk.StringVar(value="1080")  # Good balance of quality vs speed/rate limits
        self.parallel_var = tk.BooleanVar(value=True)
        self.detect_watermarks_var = tk.BooleanVar(value=True)
        self.watermark_threshold_var = tk.DoubleVar(value=0.8)
        self.fast_scene_var = tk.BooleanVar(value=False)
        self.resume_var = tk.BooleanVar(value=False)
        self.thumbnail_var = tk.BooleanVar(value=True)
        self.png_var = tk.BooleanVar(value=False)
        self.verbose_var = tk.BooleanVar(value=False)
        self.gradfun_var = tk.BooleanVar(value=False)
        self.deblock_var = tk.BooleanVar(value=True)
        self.deband_var = tk.BooleanVar(value=False)
        # YouTube authentication options
        self.cookies_from_browser_var = tk.StringVar(value="")
        self.cookies_file_var = tk.StringVar(value="")
        self.sleep_requests_var = tk.IntVar(value=0)

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
                                  values=["", "480", "720", "1080", "1440", "2160"])
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
        self.quality_value_label = ttk.Label(q_frame, text="50", width=3)
        self.quality_value_label.pack(side="right")
        ttk.Scale(q_frame, from_=0, to=100, variable=self.quality_var,
                 command=lambda v: self.quality_value_label.config(text=f"{float(v):.0f}")
                 ).pack(side="left", fill="x", expand=True, padx=4)
        ToolTip(q_frame, "Min quality 0-100. Higher = stricter. Recommended: 30-50")

        # Blur slider
        b_frame = ttk.Frame(self.main_frame)
        b_frame.pack(fill="x", pady=2)
        ttk.Label(b_frame, text="Blur:", width=8).pack(side="left")
        self.blur_value_label = ttk.Label(b_frame, text="100", width=3)
        self.blur_value_label.pack(side="right")
        ttk.Scale(b_frame, from_=0, to=500, variable=self.blur_var,
                 command=lambda v: self.blur_value_label.config(text=f"{float(v):.0f}")
                 ).pack(side="left", fill="x", expand=True, padx=4)
        ToolTip(b_frame, "Min sharpness. Higher = less blur allowed. Recommended: 50-150")

    def _create_options_section(self):
        self._create_section_header(self.main_frame, "Options")

        # Row 1: Performance
        row1 = ttk.Frame(self.main_frame)
        row1.pack(fill="x", pady=2)
        par_cb = ttk.Checkbutton(row1, text="Parallel processing", variable=self.parallel_var)
        par_cb.pack(side="left")
        ToolTip(par_cb, "Process multiple frames simultaneously.")

        # Row 2: Features
        row2 = ttk.Frame(self.main_frame)
        row2.pack(fill="x", pady=2)
        wm_cb = ttk.Checkbutton(row2, text="Detect watermarks", variable=self.detect_watermarks_var)
        wm_cb.pack(side="left", padx=(0, 16))
        ToolTip(wm_cb, "Mark frames with potential watermarks.")

        th_cb = ttk.Checkbutton(row2, text="Thumbnail", variable=self.thumbnail_var)
        th_cb.pack(side="left", padx=(0, 16))
        ToolTip(th_cb, "Generate 3x3 preview montage.")

        res_cb = ttk.Checkbutton(row2, text="Resume", variable=self.resume_var)
        res_cb.pack(side="left")
        ToolTip(res_cb, "Continue from previous extraction.")

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

    def _create_action_section(self):
        frame = ttk.Frame(self.main_frame)
        frame.pack(fill="x", pady=(12, 4))

        verbose_cb = ttk.Checkbutton(frame, text="Verbose", variable=self.verbose_var)
        verbose_cb.pack(side="left")
        ToolTip(verbose_cb, "Show detailed yt-dlp output (noisy).")

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(side="right")

        ttk.Button(btn_frame, text="Dry Run", command=self._dry_run, width=10).pack(side="left", padx=(0, 8))
        ttk.Button(btn_frame, text="Extract", style="Run.TButton", command=self._run, width=10).pack(side="left")

    def _create_status_bar(self):
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief="sunken",
                               anchor="w", font=("Segoe UI", 9), padding=(4, 2))
        status_bar.pack(side="bottom", fill="x")

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

    def _build_command(self, dry_run=False):
        source = self.source_var.get().strip()
        if not source:
            messagebox.showerror("Error", "Please enter a video URL or select a local file.")
            return None

        cmd = [sys.executable, "youtube-screenshot-script.py", source]
        method = self.method_var.get()
        cmd.extend(["--method", method])

        if method == "interval":
            cmd.extend(["--interval", str(self.interval_var.get())])
        if method == "scene" and self.fast_scene_var.get():
            cmd.append("--fast-scene")

        cmd.extend(["--quality", str(self.quality_var.get())])
        cmd.extend(["--blur", str(self.blur_var.get())])

        output = self.output_var.get().strip()
        if output:
            cmd.extend(["--output", output])

        max_res = self.max_resolution_var.get().strip()
        if max_res:
            cmd.extend(["--max-resolution", max_res])

        if self.png_var.get():
            cmd.append("--png")
        if not self.parallel_var.get():
            cmd.append("--disable-parallel")
        if self.detect_watermarks_var.get():
            cmd.append("--detect-watermarks")
            cmd.extend(["--watermark-threshold", str(self.watermark_threshold_var.get())])
        if self.thumbnail_var.get():
            cmd.append("--thumbnail")
        if self.resume_var.get():
            cmd.append("--resume")
        if self.gradfun_var.get():
            cmd.append("--gradfun")
        if self.deblock_var.get():
            cmd.append("--deblock")
        if self.deband_var.get():
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

        sleep_requests = self.sleep_requests_var.get()
        if sleep_requests > 0:
            cmd.extend(["--sleep-requests", str(sleep_requests)])

        return cmd

    def _run_command(self, cmd, dry_run=False):
        def run():
            try:
                self.status_var.set("Processing..." if not dry_run else "Dry run...")
                script_dir = os.path.dirname(os.path.abspath(__file__))
                
                # Show output window immediately in main thread
                self.root.after(0, lambda: self._create_output_window(dry_run, cmd))
                
                # Create process with unbuffered output
                env = os.environ.copy()
                env['PYTHONUNBUFFERED'] = '1'
                
                process = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, cwd=script_dir, env=env,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                    bufsize=1  # Line buffered
                )
                
                # Start reading output
                self.root.after(0, lambda: self._start_output_reader(process, dry_run))
            except Exception as e:
                self.status_var.set("Error")
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))

        threading.Thread(target=run, daemon=True).start()
    
    def _create_output_window(self, dry_run=False, cmd=None):
        """Create or reuse the output window."""
        # Reuse existing window if it exists and is still open
        if self.output_window is not None and self.output_window.winfo_exists():
            # Add separator for new run
            self._append_output(self.output_text, "\n" + "="*60 + "\n")
            self._append_output(self.output_text, f"{'Dry Run' if dry_run else 'New Extraction'} Started\n")
            self._append_output(self.output_text, "="*60 + "\n\n")
            if cmd:
                self._append_output(self.output_text, f"Command: {' '.join(cmd[:5])}...\n")
                self._append_output(self.output_text, f"Full command: {' '.join(cmd)}\n\n")
            self._append_output(self.output_text, "Processing... please wait.\n\n")
            self.output_window.lift()  # Bring to front
        else:
            # Create new window
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

            def bind_scroll(e):
                self.output_text.bind_all("<MouseWheel>", lambda ev: self.output_text.yview_scroll(int(-1*(ev.delta/120)), "units"))
            def unbind_scroll(e):
                self.output_text.unbind_all("<MouseWheel>")
            self.output_text.bind("<Enter>", bind_scroll)
            self.output_text.bind("<Leave>", unbind_scroll)

            def on_close():
                self.output_text.unbind_all("<MouseWheel>")
                self.output_window.destroy()
                self.output_window = None
                self.output_text = None
            self.output_window.protocol("WM_DELETE_WINDOW", on_close)

            ttk.Button(self.output_window, text="Close", command=on_close).pack(pady=6)

            # Initial message with command info
            if cmd:
                self._append_output(self.output_text, f"Command: {' '.join(cmd[:5])}...\n")
                self._append_output(self.output_text, f"Full command: {' '.join(cmd)}\n\n")
            self._append_output(self.output_text, "Processing... please wait.\n\n")
    
    def _start_output_reader(self, process, dry_run=False):
        """Start the output reading thread."""
        def read_output():
            try:
                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break
                    if self.output_text and self.output_text.winfo_exists():
                        self.root.after(0, lambda l=line: self._append_output(self.output_text, l))
                
                process.wait()
                status = "Complete!" if process.returncode == 0 else f"Exit code: {process.returncode}"
                self.root.after(0, lambda: self.status_var.set(status))
                if self.output_text and self.output_text.winfo_exists():
                    self.root.after(0, lambda: self._append_output(self.output_text, f"\n--- {status} ---\n"))
            except Exception as e:
                if self.output_text and self.output_text.winfo_exists():
                    self.root.after(0, lambda: self._append_output(self.output_text, f"\nError: {e}\n"))

        threading.Thread(target=read_output, daemon=True).start()

    def _append_output(self, widget, line):
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
    YouTubeScreenshotGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
