"""Unit tests for the pure helpers in youtube-screenshot-script.py.

The module name has hyphens, so it cannot be imported normally - load it by
path. Importing it pulls in cv2/numpy/PIL/yt_dlp, which is deliberate: it also
catches an import-time break against a newly resolved dependency set.
"""

import importlib.util
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
    def test_tiles_frames_of_differing_sizes(self, tmp_path):
        """remove_black_bars() crops each frame to its own content, so the
        montage must not assume every frame shares the first frame's size."""
        rng = np.random.default_rng(2)
        for i, (h, w) in enumerate([(200, 300), (120, 300), (200, 180), (50, 50)]):
            arr = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
            Image.fromarray(arr).save(tmp_path / f"frame_{i:06d}_q50_b50.jpg")

        yse.generate_thumbnail(str(tmp_path))

        montage = tmp_path / "thumbnail_montage.jpg"
        assert montage.exists()
        with Image.open(montage) as im:
            # 4 frames -> 3 columns x 2 rows of the largest cell (300x200).
            assert im.size == (900, 400)

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
