"""
Integration tests for the full pipeline.

Tests end-to-end detection on synthetic micrographs with known blobs.
Blobs use cryo-EM convention: dark (amplitude=-1) on a lighter background.
"""
from __future__ import annotations

import numpy as np
import pytest

from lipopick import PickerConfig, pick_micrograph


FACTOR = 2.0 * 2.0 ** 0.5   # sigma → diameter = sigma * FACTOR


def find_match(picks, cx, cy, tolerance_px=40.0):
    """Return the pick closest to (cx, cy), or None if none within tolerance."""
    if picks.shape[0] == 0:
        return None
    dists = np.hypot(picks[:, 0] - cx, picks[:, 1] - cy)
    idx = int(np.argmin(dists))
    if dists[idx] <= tolerance_px:
        return picks[idx]
    return None


def _make_cfg(**kwargs):
    """Return a PickerConfig with outputs disabled (test-only)."""
    defaults = dict(
        write_csv=False, write_overlay=False,
        write_histogram=False, write_extraction_plan=False,
        max_local_contrast=0.0,  # disable for synthetic blobs (unrealistic contrast)
    )
    defaults.update(kwargs)
    return PickerConfig(**defaults)


def test_pipeline_detects_small_blob(small_micrograph):
    """All three blobs in the 1024×1024 fixture should be detected."""
    image, blobs = small_micrograph
    cfg = _make_cfg(dmin=150, dmax=400, threshold_percentile=99.0)
    picks = pick_micrograph(image, cfg)
    assert picks.shape[0] > 0, "No picks produced"

    for cx, cy, diameter in blobs:
        match = find_match(picks, cx, cy, tolerance_px=40.0)
        assert match is not None, f"Blob at ({cx},{cy}) d={diameter}px not detected"


def test_pipeline_diameter_accuracy(small_micrograph):
    """Detected diameters should be within 25% of true diameters."""
    image, blobs = small_micrograph
    cfg = _make_cfg(dmin=150, dmax=400, threshold_percentile=99.0)
    picks = pick_micrograph(image, cfg)

    for cx, cy, true_d in blobs:
        match = find_match(picks, cx, cy, tolerance_px=40.0)
        if match is None:
            continue
        detected_d = float(match[2]) * FACTOR
        error = abs(detected_d - true_d) / true_d
        assert error < 0.25, (
            f"Blob d={true_d}px: detected d={detected_d:.1f}px, error={error*100:.1f}%"
        )


def test_pipeline_no_duplicates(small_micrograph):
    """Each blob should have at most 1-2 picks within its radius (NMS working)."""
    image, blobs = small_micrograph
    cfg = _make_cfg(dmin=150, dmax=400, threshold_percentile=98.0)
    picks = pick_micrograph(image, cfg)

    for cx, cy, diameter in blobs:
        radius = diameter / 2.0
        dists = np.hypot(picks[:, 0] - cx, picks[:, 1] - cy)
        n_nearby = int((dists < radius).sum())
        assert n_nearby <= 2, (
            f"Too many picks ({n_nearby}) near blob at ({cx},{cy}); NMS may be broken"
        )


def test_pipeline_large_blobs(large_micrograph):
    """Multi-scale test: detect blobs at three different pyramid levels."""
    image, blobs = large_micrograph
    cfg = _make_cfg(dmin=150, dmax=500, threshold_percentile=99.0)
    picks = pick_micrograph(image, cfg)
    assert picks.shape[0] > 0

    for cx, cy, diameter in blobs:
        match = find_match(picks, cx, cy, tolerance_px=60.0)
        assert match is not None, (
            f"Blob at ({cx},{cy}) d={diameter}px not detected in large_micrograph"
        )


def test_pipeline_separated_blobs_both_kept(packed_micrograph):
    """Two well-separated same-size blobs should both be detected."""
    image, blobs = packed_micrograph
    cfg = _make_cfg(dmin=150, dmax=300, threshold_percentile=99.0)
    picks = pick_micrograph(image, cfg)
    assert picks.shape[0] >= 2, f"Expected ≥2 picks, got {picks.shape[0]}"


def test_pipeline_empty_image():
    """Pure noise image should return few or no picks."""
    rng = np.random.default_rng(0)
    image = rng.normal(0, 0.01, size=(256, 256)).astype(np.float32)
    cfg = _make_cfg(dmin=150, dmax=300, threshold_percentile=99.9)
    picks = pick_micrograph(image, cfg)
    assert picks.shape[0] < 10, f"Too many spurious picks on noise image: {picks.shape[0]}"


# ------------------------------------------------------------------ #
# Circle overlap fraction
# ------------------------------------------------------------------ #
from lipopick.pipeline import _circle_overlap_fraction, _filter_overlapping_picks


class TestCircleOverlapFraction:
    """Unit tests for the analytical circle–circle overlap function."""

    def test_identical_circles(self):
        """Two identical circles at the same center → 100% overlap."""
        assert _circle_overlap_fraction(10.0, 10.0, 0.0) == pytest.approx(1.0)

    def test_no_overlap(self):
        """Circles far apart → 0% overlap."""
        assert _circle_overlap_fraction(5.0, 5.0, 100.0) == pytest.approx(0.0)

    def test_just_touching(self):
        """Circles exactly touching at one point → 0% overlap."""
        assert _circle_overlap_fraction(5.0, 5.0, 10.0) == pytest.approx(0.0)

    def test_containment(self):
        """Small circle fully inside large circle → 100% of smaller."""
        assert _circle_overlap_fraction(3.0, 10.0, 2.0) == pytest.approx(1.0)

    def test_partial_overlap_symmetric(self):
        """Two equal circles at half-radius separation → known partial overlap."""
        frac = _circle_overlap_fraction(10.0, 10.0, 10.0)
        assert 0.0 < frac < 1.0
        # For equal circles at d=R: overlap area / circle area ≈ 0.39
        assert frac == pytest.approx(0.39, abs=0.05)


# ------------------------------------------------------------------ #
# Post-refinement overlap filter
# ------------------------------------------------------------------ #


class TestOverlapFilter:
    """Tests for the greedy post-refinement overlap filter."""

    @staticmethod
    def _make_picks(entries):
        """Build a (N, 5) array from list of (x, y, diameter, score) tuples."""
        rows = []
        for x, y, d, score in entries:
            sigma = d / FACTOR
            rows.append([x, y, sigma, score, 1.0])
        return np.array(rows, dtype=np.float32)

    def test_removes_overlapping(self):
        """Lower-score pick overlapping a higher-score pick is removed."""
        # Two picks at same position, different scores
        picks = self._make_picks([
            (100, 100, 60.0, 0.5),   # higher score → kept
            (110, 100, 60.0, 0.3),   # overlapping, lower score → removed
        ])
        result = _filter_overlapping_picks(picks, max_overlap=0.3)
        assert result.shape[0] == 1
        assert result[0, 3] == pytest.approx(0.5)  # kept the higher score

    def test_keeps_separated(self):
        """Two far-apart picks both survive."""
        picks = self._make_picks([
            (100, 100, 60.0, 0.5),
            (500, 500, 60.0, 0.3),
        ])
        result = _filter_overlapping_picks(picks, max_overlap=0.3)
        assert result.shape[0] == 2

    def test_disabled_when_zero(self):
        """max_overlap=0 disables the filter (all picks kept)."""
        picks = self._make_picks([
            (100, 100, 60.0, 0.5),
            (110, 100, 60.0, 0.3),
        ])
        result = _filter_overlapping_picks(picks, max_overlap=0.0)
        assert result.shape[0] == 2

    def test_single_pick(self):
        """Single pick always survives."""
        picks = self._make_picks([(200, 200, 80.0, 0.4)])
        result = _filter_overlapping_picks(picks, max_overlap=0.3)
        assert result.shape[0] == 1
