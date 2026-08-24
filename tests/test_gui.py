"""Behavioural tests for the GUI, driven against a real Tk instance.

The rest of the suite checks the GUI by scanning its source, because tkinter
is not importable in every sandbox. That catches drift in flag names but not
in behaviour - a control can be drawn, wired, and still do nothing. These
tests build the real widget tree and skip cleanly where Tk cannot open a
display (headless CI runs them under xvfb).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

tk = pytest.importorskip("tkinter", reason="tkinter not installed")


def _load_gui():
    path = REPO_ROOT / "youtube-screenshot-gui.py"
    spec = importlib.util.spec_from_file_location("yse_gui", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["yse_gui"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def gui_module():
    return _load_gui()


@pytest.fixture(scope="module")
def tk_root():
    """One interpreter for the whole module.

    Creating and destroying a tk.Tk() per test is what a naive fixture does,
    but repeated interpreter teardown is flaky on Windows - Tcl starts
    failing to locate init.tcl part-way through the run. Each test gets a
    fresh Toplevel off this single root instead, which the GUI class accepts
    in place of a root window.
    """
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"No display available for Tk: {e}")
    root.withdraw()
    yield root
    root.destroy()


def _make_app(gui_module, tk_root):
    window = tk.Toplevel(tk_root)
    window.geometry("660x660")
    instance = gui_module.YouTubeScreenshotGUI(window)
    instance.source_var.set("sample.mp4")
    window.update()
    return instance


@pytest.fixture
def app(gui_module, tk_root):
    instance = _make_app(gui_module, tk_root)
    yield instance
    instance.root.destroy()


@pytest.fixture
def no_ffmpeg_app(gui_module, tk_root, monkeypatch):
    """An app built as it would be on a machine with no FFmpeg installed."""
    monkeypatch.setattr(gui_module, "check_ffmpeg", lambda: False)
    instance = _make_app(gui_module, tk_root)
    assert instance.ffmpeg_available is False
    yield instance
    instance.root.destroy()


def _select(app, method):
    app.method_var.set(method)
    app._on_method_change()
    app.root.update()


class TestKeyframesGating:
    """keyframes shells straight to FFmpeg and never runs the per-frame
    pipeline, so every control feeding that pipeline must be visibly
    unavailable AND absent from the command line - not accepted and ignored."""

    PIPELINE_FLAGS = {
        "--quality", "--blur", "--disable-parallel", "--detect-watermarks",
        "--watermark-threshold", "--resume", "--gradfun", "--deblock", "--deband",
    }

    def test_controls_are_disabled_for_keyframes(self, app):
        _select(app, "keyframes")
        for widget, _ in app._pipeline_widgets:
            assert str(widget.cget("state")) == "disabled", widget

    def test_controls_come_back_for_other_methods(self, app):
        _select(app, "keyframes")
        _select(app, "scene")
        for widget, needs_ffmpeg in app._pipeline_widgets:
            if needs_ffmpeg and not app.ffmpeg_available:
                continue
            assert str(widget.cget("state")) == "normal", widget

    def test_ffmpeg_gated_filters_stay_disabled_without_ffmpeg(self, no_ffmpeg_app):
        # Built without FFmpeg, so gradfun/deband start disabled. Switching
        # away from keyframes must not resurrect them: the method changed,
        # but the binary they need is still missing.
        _select(no_ffmpeg_app, "keyframes")
        _select(no_ffmpeg_app, "interval")
        for widget, needs_ffmpeg in no_ffmpeg_app._pipeline_widgets:
            expected = "disabled" if needs_ffmpeg else "normal"
            assert str(widget.cget("state")) == expected, widget

    def test_keyframes_command_omits_pipeline_flags(self, app):
        # Turn everything on first, so an omission is the gating and not a default.
        for var in (app.detect_watermarks_var, app.resume_var, app.gradfun_var,
                    app.deblock_var, app.deband_var):
            var.set(True)
        app.parallel_var.set(False)
        _select(app, "keyframes")
        cmd = app._build_command()
        assert cmd is not None
        assert self.PIPELINE_FLAGS.isdisjoint(cmd)
        # Options keyframes really does honour must survive.
        assert "--method" in cmd and "keyframes" in cmd

    def test_other_methods_still_send_pipeline_flags(self, app):
        app.detect_watermarks_var.set(True)
        _select(app, "scene")
        cmd = app._build_command()
        assert {"--quality", "--blur", "--detect-watermarks"} <= set(cmd)

    def test_note_is_shown_only_for_keyframes(self, app):
        _select(app, "keyframes")
        assert app.keyframes_note.winfo_ismapped()
        _select(app, "interval")
        assert not app.keyframes_note.winfo_ismapped()


class TestNumericEntryValidation:
    """A freely-typed Spinbox can hold text that DoubleVar/IntVar.get()
    rejects. Uncaught, the TclError escapes the button callback into Tk's
    error handler - which under pythonw has nowhere to print - and the button
    simply looks dead."""

    @staticmethod
    def _type_into(app, frame, text):
        from tkinter import ttk
        for widget in frame.winfo_children():
            if isinstance(widget, ttk.Spinbox):
                widget.delete(0, "end")
                widget.insert(0, text)
                app.root.update()
                return widget
        raise AssertionError("no Spinbox in frame")

    @pytest.fixture(autouse=True)
    def _capture_errors(self, app, gui_module, monkeypatch):
        self.errors = []
        monkeypatch.setattr(gui_module.messagebox, "showerror",
                            lambda title, msg: self.errors.append(msg))

    def test_non_numeric_interval_reports_instead_of_raising(self, app):
        _select(app, "interval")
        self._type_into(app, app.interval_frame, "abc")
        assert app._build_command() is None
        assert self.errors and "Interval" in self.errors[0]

    def test_out_of_range_interval_is_rejected(self, app):
        _select(app, "interval")
        self._type_into(app, app.interval_frame, "0")
        assert app._build_command() is None
        assert self.errors

    def test_valid_interval_builds_a_command(self, app):
        _select(app, "interval")
        self._type_into(app, app.interval_frame, "2.5")
        cmd = app._build_command()
        assert cmd is not None and "2.5" in cmd
        assert not self.errors

    def test_non_numeric_watermark_threshold_is_reported(self, app):
        _select(app, "scene")
        app.detect_watermarks_var.set(True)
        app.watermark_spin.delete(0, "end")
        app.watermark_spin.insert(0, "high")
        app.root.update()
        assert app._build_command() is None
        assert self.errors and "Watermark" in self.errors[0]


class TestStopBeforeProcessExists:
    """Stop pressed before Popen returns used to be dropped silently: _stop
    found self.process still None and returned without recording anything."""

    def test_stop_records_intent_with_no_process(self, app):
        app.process = None
        app._stop()
        assert app.stop_requested is True


class TestNoteWrapping:
    """The note has no horizontal scrollbar to reveal a clipped tail, so it
    must re-wrap to whatever width the window actually has."""

    @pytest.mark.parametrize("width", [480, 660, 900])
    def test_note_fits_its_width(self, app, width):
        _select(app, "keyframes")
        app.root.geometry(f"{width}x660")
        for _ in range(3):
            app.root.update()
            app.root.update_idletasks()
        note = app.keyframes_note
        assert note.winfo_reqwidth() <= note.winfo_width()
