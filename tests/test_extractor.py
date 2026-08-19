"""Unit tests for the pure helpers in youtube-screenshot-script.py.

The module name has hyphens, so it cannot be imported normally - load it by
path. Importing it pulls in cv2/numpy/PIL/yt_dlp, which is deliberate: it also
catches an import-time break against a newly resolved dependency set.
"""

import argparse
import importlib.util
import re
import io as _io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_module():
    path = REPO_ROOT / "youtube-screenshot-script.py"
    spec = importlib.util.spec_from_file_location("yse_script", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["yse_script"] = module
    spec.loader.exec_module(module)
    return module


yse = _load_module()


class TestCleanYoutubeUrl:
    @pytest.mark.parametrize(
        "url,expected",
        [
            (
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PL123",
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            ),
            (
                "https://youtu.be/dQw4w9WgXcQ?list=PL123",
                "https://youtu.be/dQw4w9WgXcQ",
            ),
            (
                "https://www.youtube.com/shorts/dQw4w9WgXcQ",
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            ),
            (
                "https://www.youtube.com/embed/dQw4w9WgXcQ",
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            ),
        ],
    )
    def test_strips_playlist_and_normalises(self, url, expected):
        assert yse.clean_youtube_url(url) == expected

    def test_non_youtube_url_untouched(self):
        url = "https://vimeo.com/12345?foo=bar"
        assert yse.clean_youtube_url(url) == url

    def test_handles_empty_and_non_string(self):
        assert yse.clean_youtube_url("") == ""
        assert yse.clean_youtube_url(None) is None


class TestParseExtractorArgs:
    def test_single_arg(self):
        assert yse.parse_extractor_args(["youtube:player_client=mweb"]) == {
            "youtube": {"player_client": ["mweb"]}
        }

    def test_multiple_values_and_args(self):
        assert yse.parse_extractor_args(
            ["youtube:player_client=mweb,web;formats=dashy"]
        ) == {"youtube": {"player_client": ["mweb", "web"], "formats": ["dashy"]}}

    def test_repeated_flag_merges_into_one_extractor(self):
        parsed = yse.parse_extractor_args(
            ["youtube:player_client=mweb", "youtube:formats=dashy"]
        )
        assert parsed == {"youtube": {"player_client": ["mweb"], "formats": ["dashy"]}}

    def test_bare_flag_gets_empty_list(self):
        assert yse.parse_extractor_args(["youtube:innertube_host"]) == {
            "youtube": {"innertube_host": []}
        }

    def test_missing_prefix_raises(self):
        with pytest.raises(ValueError):
            yse.parse_extractor_args(["player_client=mweb"])


class TestSanitize:
    def test_filename_replaces_unsafe_characters(self):
        assert yse.sanitize_filename('a/b:c*d?"e') == "a_b_c_d__e"

    def test_output_path_preserves_directory_structure(self):
        assert yse.sanitize_output_path(os.path.join("a", "b", "c")) == os.path.join(
            "a", "b", "c"
        )

    def test_output_path_sanitizes_bare_folder_name(self):
        assert yse.sanitize_output_path("my shots!") == "my_shots_"

    def test_output_path_passes_through_empty(self):
        assert yse.sanitize_output_path("") == ""


class TestRemoveBlackBars:
    def test_crops_letterbox(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        frame[20:80, :, :] = 200
        assert yse.remove_black_bars(frame).shape == (60, 100, 3)

    def test_crops_pillarbox(self):
        frame = np.zeros((100, 100, 3), np.uint8)
        frame[:, 30:70, :] = 200
        assert yse.remove_black_bars(frame).shape == (100, 40, 3)

    def test_all_black_frame_returned_unchanged(self):
        frame = np.zeros((40, 60, 3), np.uint8)
        assert yse.remove_black_bars(frame).shape == (40, 60, 3)

    def test_frame_without_bars_untouched(self):
        frame = np.full((40, 60, 3), 128, np.uint8)
        assert yse.remove_black_bars(frame).shape == (40, 60, 3)


class TestProgressTracker:
    def test_out_of_order_completion_only_advances_contiguous_prefix(self):
        tracker = yse._ProgressTracker()
        tracker.record(2, True)
        tracker.record(1, True)
        # Frame 0 is still in flight, so nothing may be considered done yet.
        assert tracker.resume_point == 0
        tracker.record(0, True)
        assert tracker.resume_point == 3
        assert tracker.saved == 3

    def test_saved_and_skipped_tallies(self):
        tracker = yse._ProgressTracker()
        for i, saved in enumerate([True, False, True, False]):
            tracker.record(i, saved)
        assert (tracker.saved, tracker.skipped, tracker.processed) == (2, 2, 4)

    def test_resumes_from_existing_counts(self):
        tracker = yse._ProgressTracker(start_at=5, saved=3, skipped=2)
        tracker.record(5, True)
        assert (tracker.resume_point, tracker.saved, tracker.skipped) == (6, 4, 2)


class TestQualityScore:
    def test_score_within_bounds(self):
        rng = np.random.default_rng(0)
        noisy = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        flat = np.full((64, 64, 3), 128, np.uint8)
        for frame in (noisy, flat):
            assert 0 <= yse.calculate_quality_score(frame) <= 100

    def test_detailed_frame_scores_above_flat_frame(self):
        rng = np.random.default_rng(1)
        noisy = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        flat = np.full((64, 64, 3), 128, np.uint8)
        assert yse.calculate_quality_score(noisy) > yse.calculate_quality_score(flat)


class TestGenerateThumbnail:
    # Distinct, well-separated colours so each tile is identifiable in the montage.
    TILE_SIZES = [(200, 300), (120, 300), (200, 180), (50, 50)]
    TILE_COLOURS = [(220, 20, 20), (20, 220, 20), (20, 20, 220), (220, 220, 20)]

    def _write_frames(self, tmp_path):
        for i, ((h, w), colour) in enumerate(zip(self.TILE_SIZES, self.TILE_COLOURS)):
            arr = np.zeros((h, w, 3), np.uint8)
            arr[:, :] = colour
            # PNG so the source tiles carry no JPEG ringing of their own.
            Image.fromarray(arr).save(tmp_path / f"frame_{i:06d}_q50_b50.png")

    def test_montage_is_a_grid_of_the_largest_cell(self, tmp_path):
        self._write_frames(tmp_path)
        yse.generate_thumbnail(str(tmp_path))
        with Image.open(tmp_path / "thumbnail_montage.jpg") as im:
            # 4 frames -> 3 columns x 2 rows of the largest cell (300x200).
            assert im.size == (900, 400)

    def test_every_cell_centre_holds_its_own_frame(self, tmp_path):
        """The real defect was placement, not overall size: tiles were pasted at
        their own dimensions into cells sized from the FIRST frame, so anything
        smaller left black gaps and anything larger overlapped its neighbour.
        Checking the centre of each cell is what distinguishes the two."""
        self._write_frames(tmp_path)
        yse.generate_thumbnail(str(tmp_path))

        cell_w, cell_h, cols = 300, 200, 3
        with Image.open(tmp_path / "thumbnail_montage.jpg") as im:
            montage = im.convert("RGB")
            for i, expected in enumerate(self.TILE_COLOURS):
                x = (i % cols) * cell_w + cell_w // 2
                y = (i // cols) * cell_h + cell_h // 2
                actual = montage.getpixel((x, y))
                # Generous tolerance: the montage itself is saved as JPEG.
                assert all(abs(a - e) <= 40 for a, e in zip(actual, expected)), (
                    f"cell {i} centre is {actual}, expected roughly {expected}"
                )

    def test_tiles_keep_their_aspect_ratio(self, tmp_path):
        """A 50x50 frame must not be stretched to fill a 300x200 cell."""
        self._write_frames(tmp_path)
        yse.generate_thumbnail(str(tmp_path))
        cell_w, cell_h, cols = 300, 200, 3
        with Image.open(tmp_path / "thumbnail_montage.jpg") as im:
            montage = im.convert("RGB")
            # Frame 3 is the square one; its cell is wider than it is tall, so
            # scaled-to-fit leaves background either side of a 200x200 tile.
            x0 = (3 % cols) * cell_w
            y = (3 // cols) * cell_h + cell_h // 2
            assert montage.getpixel((x0 + 2, y)) != self.TILE_COLOURS[3]
            assert all(
                abs(a - e) <= 40
                for a, e in zip(montage.getpixel((x0 + cell_w // 2, y)),
                                self.TILE_COLOURS[3])
            )

    def test_no_frames_is_a_no_op(self, tmp_path):
        yse.generate_thumbnail(str(tmp_path))
        assert not (tmp_path / "thumbnail_montage.jpg").exists()

    def test_excludes_previous_montage(self, tmp_path):
        arr = np.full((20, 20, 3), 100, np.uint8)
        Image.fromarray(arr).save(tmp_path / "frame_000000_q50_b50.jpg")
        Image.fromarray(arr).save(tmp_path / "thumbnail_montage.jpg")
        yse.generate_thumbnail(str(tmp_path))
        with Image.open(tmp_path / "thumbnail_montage.jpg") as im:
            assert im.size == (20, 20)


class TestBuildYdlOpts:
    def test_cookies_from_browser_is_a_tuple(self):
        opts = yse.build_ydl_opts("out.mp4", cookies_from_browser="firefox")
        assert opts["cookiesfrombrowser"] == ("firefox",)

    def test_cookie_file_used_when_no_browser(self):
        opts = yse.build_ydl_opts("out.mp4", cookies_file="cookies.txt")
        assert opts["cookiefile"] == "cookies.txt"

    def test_browser_cookies_win_over_file(self):
        opts = yse.build_ydl_opts(
            "out.mp4", cookies_from_browser="chrome", cookies_file="cookies.txt"
        )
        assert "cookiefile" not in opts

    def test_max_resolution_appears_in_format_string(self):
        opts = yse.build_ydl_opts("out.mp4", max_resolution=720)
        assert "height<=720" in opts["format"]

    def test_max_resolution_survives_the_no_ffmpeg_fallback(self, monkeypatch):
        """Without FFmpeg the format string is swapped for a video-only one; it
        must still cap the resolution rather than grabbing the largest stream."""
        monkeypatch.setattr(yse, "check_ffmpeg", lambda: False)
        opts = yse.build_ydl_opts("out.mp4", max_resolution=720)
        assert "height<=720" in opts["format"]
        assert "+bestaudio" not in opts["format"]

    def test_no_ffmpeg_fallback_without_cap_is_unconstrained(self, monkeypatch):
        monkeypatch.setattr(yse, "check_ffmpeg", lambda: False)
        assert yse.build_ydl_opts("out.mp4")["format"] == "bestvideo/best"

    def test_sleep_requests_maps_to_yt_dlp_option(self):
        opts = yse.build_ydl_opts("out.mp4", sleep_requests=5)
        assert opts["sleep_interval_requests"] == 5

    def test_sleep_requests_omitted_when_zero(self):
        assert "sleep_interval_requests" not in yse.build_ydl_opts("out.mp4")


class TestLoadProgress:
    """--resume exists for crashed runs, so a half-written progress file is the
    expected input, not an exotic one."""

    def test_valid_file_is_honoured(self, tmp_path):
        f = tmp_path / "progress.json"
        f.write_text(
            json.dumps({"processed_frames": 7, "skipped_frames": 2, "saved_frames": 5})
        )
        assert yse.load_progress(str(f)) == (7, 2, 5)

    def test_truncated_file_restarts_instead_of_raising(self, tmp_path):
        f = tmp_path / "progress.json"
        f.write_text('{"processed_frames": 1, "skipped_')
        assert yse.load_progress(str(f)) == (0, 0, 0)

    def test_empty_file_restarts(self, tmp_path):
        f = tmp_path / "progress.json"
        f.write_text("")
        assert yse.load_progress(str(f)) == (0, 0, 0)

    def test_wrong_types_restart(self, tmp_path):
        f = tmp_path / "progress.json"
        f.write_text(json.dumps({"processed_frames": "lots"}))
        assert yse.load_progress(str(f)) == (0, 0, 0)

    def test_missing_file_restarts(self, tmp_path):
        assert yse.load_progress(str(tmp_path / "nope.json")) == (0, 0, 0)

    def test_missing_keys_default_to_zero(self, tmp_path):
        f = tmp_path / "progress.json"
        f.write_text(json.dumps({"processed_frames": 4}))
        assert yse.load_progress(str(f)) == (4, 0, 0)


class TestConfigFile:
    """--config used to be a bare set_defaults(**json.load(f)), so a typo was
    silently ignored and a wrong type surfaced as a traceback mid-extraction."""

    @staticmethod
    def _apply(tmp_path, payload):
        parser = argparse.ArgumentParser()
        parser.add_argument("source")
        parser.add_argument("--config")
        parser.add_argument("--quality", type=float, default=30.0)
        parser.add_argument("--method", choices=["interval", "scene"], default="interval")
        parser.add_argument("--png", action="store_true")
        parser.add_argument("--output", type=str)
        parser.add_argument("--max-resolution", type=int)
        parser.add_argument("--extractor-args", type=str, action="append")
        parser.add_argument("--use-gpu", action="store_true", help=argparse.SUPPRESS)

        path = tmp_path / "config.json"
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload))
        appends = yse.apply_config_file(parser, str(path))
        return parser, appends

    @classmethod
    def _parse(cls, tmp_path, payload, argv):
        """Apply a config file, then parse argv the way main() does."""
        parser, appends = cls._apply(tmp_path, payload)
        args = parser.parse_args(argv)
        for dest, value in appends.items():
            if getattr(args, dest) is None:
                setattr(args, dest, value)
        return args

    def test_valid_values_become_defaults(self, tmp_path):
        args = self._parse(
            tmp_path,
            {"quality": 42, "method": "scene", "png": True, "output": "shots"},
            ["video.mp4"],
        )
        assert (args.quality, args.method, args.png, args.output) == (
            42.0,
            "scene",
            True,
            "shots",
        )

    def test_command_line_still_wins_over_the_file(self, tmp_path):
        args = self._parse(tmp_path, {"quality": 42}, ["video.mp4", "--quality", "77"])
        assert args.quality == 77.0

    def test_hyphenated_keys_are_accepted(self, tmp_path):
        assert self._parse(tmp_path, {"max-resolution": 720}, ["video.mp4"]).max_resolution == 720

    def test_append_option_accepts_a_bare_string(self, tmp_path):
        args = self._parse(
            tmp_path, {"extractor-args": "youtube:player_client=mweb"}, ["video.mp4"]
        )
        assert args.extractor_args == ["youtube:player_client=mweb"]

    def test_command_line_replaces_rather_than_extends_an_append_option(self, tmp_path):
        """argparse's append action EXTENDS a non-None default, so routing the
        file's value through set_defaults would merge the two instead of
        letting the command line win like every other setting does."""
        args = self._parse(
            tmp_path,
            {"extractor-args": "youtube:player_client=mweb"},
            ["video.mp4", "--extractor-args", "youtube:formats=dashy"],
        )
        assert args.extractor_args == ["youtube:formats=dashy"]

    def test_repeated_command_line_flags_still_accumulate(self, tmp_path):
        args = self._parse(
            tmp_path,
            {"quality": 42},
            ["video.mp4", "--extractor-args", "youtube:a=1", "--extractor-args", "youtube:b=2"],
        )
        assert args.extractor_args == ["youtube:a=1", "youtube:b=2"]

    def test_non_integral_float_is_rejected_for_an_int_option(self, tmp_path, capsys):
        """int(5.7) would truncate silently; the CLI rejects --max-resolution 5.7."""
        with pytest.raises(SystemExit):
            self._apply(tmp_path, {"max-resolution": 720.5})
        assert "whole number" in capsys.readouterr().err

    def test_integral_float_is_accepted_for_an_int_option(self, tmp_path):
        assert self._parse(tmp_path, {"max-resolution": 720.0}, ["video.mp4"]).max_resolution == 720

    @pytest.mark.parametrize(
        "payload,expected",
        [
            ({"qualtiy": 42}, "Unknown setting"),
            ({"quality": "high"}, "must be a number"),
            ({"quality": True}, "must be a number"),
            ({"png": 1}, "must be true or false"),
            ({"method": "sideways"}, "must be one of"),
            ({"output": 5}, "must be a string"),
            ({"extractor-args": [1, 2]}, "list of strings"),
            ({"source": "video.mp4"}, "cannot be set from a config file"),
            ('{"quality": 4', "not valid JSON"),
            ("[1, 2, 3]", "must contain a JSON object"),
        ],
    )
    def test_bad_input_is_a_clean_error_not_a_traceback(
        self, tmp_path, capsys, payload, expected
    ):
        with pytest.raises(SystemExit):
            self._apply(tmp_path, payload)
        assert expected in capsys.readouterr().err

    def test_suggestions_omit_source_and_hidden_options(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            self._apply(tmp_path, {"nope": 1})
        err = capsys.readouterr().err
        assert "quality" in err
        assert "source" not in err.split("Valid settings:")[1]
        assert "use_gpu" not in err

    def test_missing_file_is_a_clean_error(self, tmp_path):
        parser = argparse.ArgumentParser()
        with pytest.raises(SystemExit):
            yse.apply_config_file(parser, str(tmp_path / "absent.json"))

    def test_returns_only_append_settings_for_the_caller(self, tmp_path):
        _, appends = self._apply(
            tmp_path, {"quality": 42, "extractor-args": "youtube:a=1"}
        )
        assert appends == {"extractor_args": ["youtube:a=1"]}


class TestStatusReporter:
    """Per-frame lines used to print unconditionally while the progress bar was
    verbose-only - backwards, since an 'all' run emits tens of thousands."""

    class _FakeOut(_io.StringIO):
        def __init__(self, tty):
            super().__init__()
            self._tty = tty

        def isatty(self):
            return self._tty

    class _Tracker:
        processed, saved, skipped = 5, 4, 1

    def _reporter(self, monkeypatch, tty, verbose):
        monkeypatch.setattr(sys, "stdout", self._FakeOut(tty))
        return yse._StatusReporter(10, 0, verbose)

    @pytest.mark.parametrize(
        "tty,verbose,bar_expected",
        [(True, False, True), (True, True, True), (False, False, False), (False, True, False)],
    )
    def test_bar_only_on_a_terminal(self, monkeypatch, tty, verbose, bar_expected):
        reporter = self._reporter(monkeypatch, tty, verbose)
        assert (not reporter._bar.disable) is bar_expected

    @pytest.mark.parametrize("tty", [True, False])
    def test_per_frame_lines_are_verbose_only(self, monkeypatch, tty):
        quiet = self._reporter(monkeypatch, tty, verbose=False)
        loud = self._reporter(monkeypatch, tty, verbose=True)
        emitted = []
        monkeypatch.setattr(yse._StatusReporter, "_emit", staticmethod(emitted.append))
        quiet.log("Saved frame 1")
        assert emitted == []
        loud.log("Saved frame 1")
        assert emitted == ["Saved frame 1"]

    def test_piped_run_gets_periodic_summaries_instead(self, monkeypatch):
        reporter = self._reporter(monkeypatch, tty=False, verbose=False)
        emitted = []
        monkeypatch.setattr(yse._StatusReporter, "_emit", staticmethod(emitted.append))
        reporter.advance(self._Tracker)
        assert emitted and "5/10 frames" in emitted[0] and "4 saved" in emitted[0]

    def test_summaries_are_rate_limited(self, monkeypatch):
        reporter = self._reporter(monkeypatch, tty=False, verbose=False)
        emitted = []
        monkeypatch.setattr(yse._StatusReporter, "_emit", staticmethod(emitted.append))
        for _ in range(50):
            reporter.advance(self._Tracker)
        # One immediate line, then nothing more inside the interval window.
        assert len(emitted) == 1

    def test_terminal_run_gets_no_summary_lines(self, monkeypatch):
        reporter = self._reporter(monkeypatch, tty=True, verbose=False)
        emitted = []
        monkeypatch.setattr(yse._StatusReporter, "_emit", staticmethod(emitted.append))
        reporter.advance(self._Tracker)
        assert emitted == []

    def test_summary_line_copes_with_unknown_total(self, monkeypatch):
        monkeypatch.setattr(sys, "stdout", self._FakeOut(False))
        reporter = yse._StatusReporter(None, 0, False)
        assert "5 frames" in reporter.summary_line(self._Tracker)


class TestProcessFrameScoring:
    """The filename's scores must describe the frame that was actually saved."""

    @staticmethod
    def _letterboxed(inner):
        frame = np.zeros((200, 200, 3), np.uint8)
        frame[70:130, :, :] = inner
        return frame

    def test_scores_are_measured_after_cropping(self):
        rng = np.random.default_rng(3)
        inner = rng.integers(0, 255, (60, 200, 3), dtype=np.uint8)
        letterboxed = self._letterboxed(inner)

        # Scoring the uncropped frame folds the black bars into contrast and
        # entropy, so it lands well below the score of the picture alone.
        assert yse.calculate_quality_score(letterboxed) < yse.calculate_quality_score(
            yse.remove_black_bars(letterboxed)
        )

    def test_saved_filename_reports_the_cropped_frames_score(self, tmp_path):
        rng = np.random.default_rng(4)
        inner = rng.integers(0, 255, (60, 200, 3), dtype=np.uint8)
        letterboxed = self._letterboxed(inner)
        expected = int(yse.calculate_quality_score(yse.remove_black_bars(letterboxed)))

        status, saved, _ = yse.process_frame(
            (letterboxed, str(tmp_path), 0, 0.0, 0.0, False, 0.8, False,
             False, False, False, False, False)
        )
        assert saved
        written = list(tmp_path.iterdir())
        assert len(written) == 1
        assert f"_q{expected:02d}_" in written[0].name

    def test_saved_image_matches_the_cropped_dimensions(self, tmp_path):
        rng = np.random.default_rng(5)
        inner = rng.integers(0, 255, (60, 200, 3), dtype=np.uint8)
        yse.process_frame(
            (self._letterboxed(inner), str(tmp_path), 0, 0.0, 0.0, False, 0.8, False,
             False, False, False, False, False)
        )
        with Image.open(next(tmp_path.iterdir())) as im:
            assert im.size == (200, 60)

    def test_low_quality_frame_is_skipped(self, tmp_path):
        flat = np.full((100, 100, 3), 128, np.uint8)
        status, saved, _ = yse.process_frame(
            (flat, str(tmp_path), 0, 99.0, 0.0, False, 0.8, False,
             False, False, False, False, False)
        )
        assert not saved
        assert "low quality" in status
        assert list(tmp_path.iterdir()) == []


class TestDefaults:
    """The CLI and the GUI disagreed on thresholds: 12/10 against 50/100."""

    def test_extract_frames_defaults_match_the_documented_pair(self):
        import inspect

        params = inspect.signature(yse.extract_frames).parameters
        assert params["quality_threshold"].default == 30
        assert params["blur_threshold"].default == 50


class TestCropFastPath:
    """remove_black_bars runs on every frame now (scores must describe the
    cropped image), so it takes a fast path when no edge is a bar. That path
    must be exactly equivalent to the full scan, not merely close."""

    @staticmethod
    def _full_scan(frame, threshold=10):
        """The unconditional implementation, kept here as the oracle."""
        black_mask = (frame[:, :, :3] < threshold).all(axis=2)
        rows = np.where(~black_mask.all(axis=1))[0]
        if rows.size == 0:
            return frame
        top, bottom = rows[0], rows[-1]
        cols = np.where(~black_mask[top:bottom + 1].all(axis=0))[0]
        if cols.size == 0:
            return frame
        return frame[top:bottom + 1, cols[0]:cols[-1] + 1]

    def test_matches_the_full_scan_on_randomised_frames(self):
        rng = np.random.default_rng(7)
        for trial in range(400):
            h, w = rng.integers(2, 50, 2)
            frame = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            shape = trial % 8
            if shape == 1:
                frame[: max(1, h // 3)] = 0
                frame[-max(1, h // 3):] = 0
            elif shape == 2:
                frame[:, : max(1, w // 3)] = 0
            elif shape == 3:
                frame[:] = 0
            elif shape == 4:
                frame = frame % 12          # values straddling the threshold
            elif shape == 5:
                frame[0] = 0                # bar on one side only
            elif shape == 6:
                frame[:, -1] = 0
            elif shape == 7:
                frame[0] = frame[-1] = 0
                frame[:, 0] = frame[:, -1] = 0

            fast = yse.remove_black_bars(frame)
            slow = self._full_scan(frame)
            assert fast.shape == slow.shape
            assert np.array_equal(fast, slow)

    def test_frame_with_content_on_every_edge_is_returned_untouched(self):
        frame = np.full((30, 40, 3), 200, np.uint8)
        assert yse.remove_black_bars(frame) is frame

    def test_a_single_dark_edge_still_triggers_the_scan(self):
        frame = np.full((30, 40, 3), 200, np.uint8)
        frame[0] = 0
        assert yse.remove_black_bars(frame).shape == (29, 40, 3)


class TestGuiCliParity:
    """The GUI shells out to the CLI, so a flag it emits that the CLI does not
    accept is a silently broken option - the failure mode this repo keeps
    hitting. Checked statically because tkinter is not importable everywhere."""

    GUI = REPO_ROOT / "youtube-screenshot-gui.py"
    CLI = REPO_ROOT / "youtube-screenshot-script.py"
    # Deliberately CLI-only: --config is a scripting convenience, --use-gpu is
    # a deprecated no-op hidden from --help.
    CLI_ONLY = {"--config", "--use-gpu"}

    def _flags(self):
        gui = set(re.findall(r'"(--[a-z0-9][a-z0-9-]*)"', self.GUI.read_text()))
        cli = set(
            re.findall(
                r'add_argument\(\s*"(--[a-z0-9][a-z0-9-]*)"', self.CLI.read_text()
            )
        )
        return gui, cli

    def test_every_flag_the_gui_emits_is_accepted_by_the_cli(self):
        gui, cli = self._flags()
        assert gui, "found no flags in the GUI - the scan pattern has drifted"
        assert gui - cli == set()

    def test_only_the_known_options_are_cli_only(self):
        gui, cli = self._flags()
        assert cli - gui == self.CLI_ONLY

    def test_thresholds_match_between_the_interfaces(self):
        gui = self.GUI.read_text()
        assert "self.quality_var = tk.DoubleVar(value=30.0)" in gui
        assert "self.blur_var = tk.DoubleVar(value=50.0)" in gui
