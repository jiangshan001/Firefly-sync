"""Tests for ROI localisation and brightness helpers.

These tests use synthetic numpy arrays — no Pi hardware required.
Runs on any platform with numpy + opencv-python installed.
"""

import numpy as np
import pytest

from firefly_sync.hardware.flash_detector import (
    compute_bright_blob_metrics,
    compute_roi_mean_brightness,
    compute_top_percentile_brightness,
    normalize_frame_to_grayscale,
)
from firefly_sync.hardware.roi_locator import (
    locate_flashing_region,
    locate_flashing_regions,
)


# ---------------------------------------------------------------------------
# Synthetic frame generators
# ---------------------------------------------------------------------------

def _make_grey(w: int, h: int, value: int) -> np.ndarray:
    """Uniform greyscale frame."""
    return np.full((h, w), value, dtype=np.uint8)


def _make_grey_with_square(
    w: int, h: int, bg: int, sq_x: int, sq_y: int,
    sq_w: int, sq_h: int, sq_val: int,
) -> np.ndarray:
    """Greyscale frame with a bright square on a dark background."""
    frame = np.full((h, w), bg, dtype=np.uint8)
    frame[sq_y:sq_y + sq_h, sq_x:sq_x + sq_w] = sq_val
    return frame


# ---------------------------------------------------------------------------
# compute_roi_mean_brightness
# ---------------------------------------------------------------------------

class TestRoiMeanBrightness:
    def test_full_frame_uniform(self) -> None:
        f = _make_grey(100, 80, 128)
        assert compute_roi_mean_brightness(f) == 128.0

    def test_roi_crop(self) -> None:
        # 200x100 bg=0 with 50x50 square of 255 at (20, 30)
        f = np.zeros((100, 200), dtype=np.uint8)
        f[30:80, 20:70] = 255
        roi = [20, 30, 50, 50]
        m = compute_roi_mean_brightness(f, roi=roi)
        assert m == 255.0

    def test_roi_partial(self) -> None:
        f = np.full((100, 100), 50, dtype=np.uint8)
        f[40:60, 40:60] = 200
        roi = [30, 30, 40, 40]
        m = compute_roi_mean_brightness(f, roi=roi)
        # 75% of roi is 50, 25% is 200 → weighted
        expected = (0.75 * 50 + 0.25 * 200)
        assert m == pytest.approx(expected, rel=0.05)

    def test_bgr_frame(self) -> None:
        f = np.full((50, 50, 3), 100, dtype=np.uint8)
        f[20:30, 20:30] = [200, 200, 200]
        m = compute_roi_mean_brightness(f)
        assert 100 < m < 120


# ---------------------------------------------------------------------------
# compute_top_percentile_brightness
# ---------------------------------------------------------------------------

class TestTopPercentileBrightness:
    def test_uniform(self) -> None:
        f = _make_grey(100, 80, 128)
        assert compute_top_percentile_brightness(f, percentile=99) == 128.0

    def test_small_bright_region(self) -> None:
        """~5% of pixels are bright (255), 95% are dark (10).  p99 must
        capture the bright pixels; p50 (median) stays dark."""
        f = np.full((100, 100), 10, dtype=np.uint8)
        f[40:60, 40:60] = 255  # 20x20 = 400 px = 4%
        p99 = compute_top_percentile_brightness(f, percentile=99.0)
        # At p99, we should be solidly inside the bright region
        assert p99 > 200

        # At p50 (median), still dark
        p50 = compute_top_percentile_brightness(f, percentile=50.0)
        assert p50 < 30

    def test_with_roi(self) -> None:
        f = np.full((100, 100), 10, dtype=np.uint8)
        f[20:40, 20:40] = 255
        roi = [15, 15, 30, 30]
        p99 = compute_top_percentile_brightness(f, percentile=99.0, roi=roi)
        assert p99 > 200

    def test_zero_percentile_is_min(self) -> None:
        f = np.array([[10, 50, 200]], dtype=np.uint8)
        assert compute_top_percentile_brightness(f, percentile=0) == 10.0

    def test_100_percentile_is_max(self) -> None:
        f = np.array([[10, 50, 200]], dtype=np.uint8)
        assert compute_top_percentile_brightness(f, percentile=100) == 200.0


# ---------------------------------------------------------------------------
# compute_bright_blob_metrics
# ---------------------------------------------------------------------------

class TestBrightBlobMetrics:
    def test_no_blob_dark_frame(self) -> None:
        f = _make_grey(100, 80, 10)
        m = compute_bright_blob_metrics(f, threshold=128)
        assert m["blob_found"] is False
        assert m["blob_area_px"] == 0

    def test_one_bright_blob(self) -> None:
        f = np.full((100, 100), 10, dtype=np.uint8)
        f[30:60, 30:60] = 200  # 30x30 = 900 px bright blob
        m = compute_bright_blob_metrics(f, threshold=128)
        assert m["blob_found"] is True
        assert m["blob_area_px"] == 900
        assert abs(m["blob_mean_brightness"] - 200.0) < 3

    def test_blob_bbox(self) -> None:
        f = np.full((100, 100), 10, dtype=np.uint8)
        f[30:55, 20:65] = 200  # 45x35 blob at (20, 30)
        m = compute_bright_blob_metrics(f, threshold=128)
        assert m["blob_found"] is True
        x, y, w, h = m["blob_bbox"]
        assert abs(x - 20) <= 1
        assert abs(y - 30) <= 1
        assert abs(w - 45) <= 1
        assert abs(h - 25) <= 1

    def test_with_roi(self) -> None:
        f = np.full((100, 100), 10, dtype=np.uint8)
        f[10:20, 10:20] = 200
        f[60:80, 60:80] = 220  # larger blob outside ROI
        roi = [5, 5, 30, 30]
        m = compute_bright_blob_metrics(f, threshold=128, roi=roi)
        assert m["blob_found"] is True
        # Should find the smaller blob in the ROI area
        assert m["blob_area_px"] <= 200  # ~10x10 = 100 px

    def test_frame_formats_are_normalized_for_blob_metrics(self) -> None:
        grey = np.full((50, 60), 10, dtype=np.uint8)
        grey[10:20, 15:25] = 240
        bgr = np.repeat(grey[:, :, None], 3, axis=2)
        bgra = np.dstack([bgr, np.full_like(grey, 255)])

        for frame in (grey, bgr, bgra):
            normalized = normalize_frame_to_grayscale(frame)
            assert normalized.ndim == 2
            assert normalized.dtype == np.uint8
            m = compute_bright_blob_metrics(frame, threshold=128)
            assert m["blob_found"] is True
            assert m["blob_area_px"] == 100


# ---------------------------------------------------------------------------
# locate_flashing_region
# ---------------------------------------------------------------------------

class TestLocateFlashingRegion:
    def test_no_frames_returns_none(self) -> None:
        assert locate_flashing_region([]) is None
        assert locate_flashing_region([_make_grey(100, 100, 128)]) is None

    def test_no_variation_returns_none(self) -> None:
        frames = [_make_grey(100, 100, 128) for _ in range(5)]
        assert locate_flashing_region(frames) is None

    def test_flashing_square_found(self) -> None:
        """Alternating bright/dark square at (20,20,30,30)."""
        frames = []
        for i in range(20):
            if i % 2 == 0:
                f = _make_grey_with_square(200, 150, 10, 50, 40, 40, 40, 250)
            else:
                f = _make_grey(200, 150, 10)
            frames.append(f)

        result = locate_flashing_region(frames, min_area_px=50, padding_px=5)
        assert result is not None, "Should find the flashing square"
        # ROI should roughly cover the square at (50,40,40,40)
        # With padding 5: (45,35,50,50) approx
        assert abs(result["x"] - 45) <= 10
        assert abs(result["y"] - 35) <= 10
        assert result["width"] >= 40
        assert result["height"] >= 40
        assert result["confidence"] > 0.1

    def test_min_area_filters_noise(self) -> None:
        """Tiny 2x2 flashing region should be rejected if min_area=100."""
        frames = []
        for i in range(10):
            if i % 2 == 0:
                f = _make_grey_with_square(200, 150, 10, 10, 10, 2, 2, 250)
            else:
                f = _make_grey(200, 150, 10)
            frames.append(f)

        result = locate_flashing_region(frames, min_area_px=100)
        assert result is None, "Tiny region should be filtered out"

    def test_padding_and_clipping(self) -> None:
        """Edge-located flashing region should not exceed frame bounds."""
        frames = []
        for i in range(20):
            if i % 2 == 0:
                f = _make_grey_with_square(200, 150, 10, 0, 0, 20, 20, 250)
            else:
                f = _make_grey(200, 150, 10)
            frames.append(f)

        result = locate_flashing_region(frames, min_area_px=30, padding_px=50)
        assert result is not None
        # Padding from top-left corner should be clipped to 0
        assert result["x"] == 0
        assert result["y"] == 0
        assert result["width"] <= 200

    def test_max_min_range_method(self) -> None:
        frames = []
        for i in range(20):
            v = 250 if i % 2 == 0 else 10
            frames.append(_make_grey_with_square(100, 100, 10, 30, 30, 20, 20, v))

        result = locate_flashing_region(frames, method="max_min_range")
        assert result is not None
        assert result["method"] == "max_min_range"

    def test_mean_abs_diff_method(self) -> None:
        frames = []
        for i in range(20):
            v = 250 if i % 2 == 0 else 10
            frames.append(_make_grey_with_square(100, 100, 10, 30, 30, 20, 20, v))

        result = locate_flashing_region(frames, method="mean_abs_diff")
        assert result is not None

    def test_downsample(self) -> None:
        frames = []
        for i in range(20):
            v = 250 if i % 2 == 0 else 10
            frames.append(_make_grey_with_square(100, 100, 10, 30, 30, 20, 20, v))

        result = locate_flashing_region(frames, downsample=2)
        assert result is not None

    def test_custom_threshold(self) -> None:
        frames = []
        for i in range(20):
            v = 250 if i % 2 == 0 else 10
            frames.append(_make_grey_with_square(100, 100, 10, 30, 30, 20, 20, v))

        result = locate_flashing_region(frames, change_threshold=30)
        assert result is not None
        assert result["auto_threshold"] == 30

    def test_bgr_frames_accepted(self) -> None:
        """BGR 3-channel frames should be converted to grey automatically."""
        frames = []
        for i in range(20):
            bg = np.full((100, 100, 3), 10, dtype=np.uint8)
            if i % 2 == 0:
                bg[30:50, 30:50] = [250, 250, 250]
            frames.append(bg)

        result = locate_flashing_region(frames, min_area_px=50)
        assert result is not None


class TestLocateFlashingRegions:
    def test_two_components_are_selected_independently(self) -> None:
        frames = []
        for i in range(30):
            f = _make_grey(220, 140, 10)
            if i % 2 == 0:
                f[40:70, 30:60] = 250
            if i % 3 == 0:
                f[45:75, 150:180] = 250
            frames.append(f)

        result = locate_flashing_regions(
            frames,
            max_regions=2,
            min_area_px=100,
            padding_px=5,
            max_overlap_ratio=0.1,
        )

        assert result["failure_reason"] is None
        assert len(result["regions"]) == 2
        rois = sorted([item["roi"] for item in result["regions"]])
        assert rois[0][0] < 50
        assert rois[1][0] > 130

    def test_overlapping_components_do_not_fill_two_slots(self) -> None:
        frames = []
        for i in range(20):
            f = _make_grey(120, 100, 10)
            if i % 2 == 0:
                f[30:70, 40:80] = 250
            frames.append(f)

        result = locate_flashing_regions(
            frames,
            max_regions=2,
            min_area_px=100,
            padding_px=10,
            max_overlap_ratio=0.1,
        )

        assert len(result["regions"]) == 1
