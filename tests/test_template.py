"""
Tests for the dark-disc template matching detector.

Tests cover:
  1. Template construction (zero-mean, shape, symmetry)
  2. NCC on a synthetic dark disc (peak at center, high score)
  3. Multi-scale best radius detection
  4. Integration with conftest fixtures via pick_micrograph(detection_method="template")
  5. Empty/noise image (few or no picks)
  6. Config validation (invalid detection_method)
"""
from __future__ import annotations

import numpy as np
import pytest

from lipopick import PickerConfig, pick_micrograph
from lipopick.template import (
    make_dark_disc_template,
    _compute_ncc,
    compute_correlation_maps,
    find_correlation_peaks,
)


FACTOR = 2.0 * 2.0 ** 0.5   # sigma → diameter = sigma * FACTOR


def _make_cfg(**kwargs):
    """Return a PickerConfig with outputs disabled (test-only)."""
    defaults = dict(
        write_csv=False, write_overlay=False,
        write_histogram=False, write_extraction_plan=False,
        max_local_contrast=0.0,  # disable for synthetic blobs (unrealistic contrast)
    )
    defaults.update(kwargs)
    return PickerConfig(**defaults)


# ------------------------------------------------------------------ #
# 1. Template construction
# ------------------------------------------------------------------ #

class TestMakeDarkDiscTemplate:

    def test_zero_mean(self):
        """Template must be zero-mean."""
        t = make_dark_disc_template(20.0)
        assert abs(t.sum()) < 1e-3, f"Template sum = {t.sum()}"

    def test_shape(self):
        """Template shape should be (2*R_out+1, 2*R_out+1) approximately."""
        r = 15.0
        t = make_dark_disc_template(r, annulus_width_frac=0.5, max_annulus_width=40.0)
        r_out = r + 0.5 * r  # annulus_width = 0.5 * radius
        expected_size = int(np.ceil(r_out)) * 2 + 1
        assert t.shape == (expected_size, expected_size)

    def test_radial_symmetry(self):
        """Template should be symmetric under 90-degree rotation."""
        t = make_dark_disc_template(25.0)
        assert np.allclose(t, np.rot90(t)), "Template not rotationally symmetric"

    def test_interior_negative(self):
        """Interior disc pixels should be negative (dark)."""
        r = 20.0
        t = make_dark_disc_template(r)
        center = t.shape[0] // 2
        assert t[center, center] < 0, "Center pixel should be negative"

    def test_annulus_positive(self):
        """Annulus pixels should be positive (bright)."""
        r = 20.0
        t = make_dark_disc_template(r, annulus_width_frac=0.5)
        center = t.shape[0] // 2
        # Check a pixel in the annulus (at radius + half annulus_width)
        annulus_pos = int(center + r + r * 0.25)
        if annulus_pos < t.shape[0]:
            assert t[center, annulus_pos] > 0, "Annulus pixel should be positive"

    def test_max_annulus_width_cap(self):
        """Annulus width should be capped by max_annulus_width."""
        r = 100.0
        t_uncapped = make_dark_disc_template(r, annulus_width_frac=0.5, max_annulus_width=200.0)
        t_capped = make_dark_disc_template(r, annulus_width_frac=0.5, max_annulus_width=20.0)
        assert t_capped.shape[0] < t_uncapped.shape[0], "Cap should produce smaller template"


# ------------------------------------------------------------------ #
# 2. NCC on synthetic dark disc
# ------------------------------------------------------------------ #

class TestComputeNCC:

    def test_peak_at_disc_center(self):
        """NCC should peak at the center of a synthetic dark disc."""
        size = 256
        image = np.random.default_rng(42).normal(0, 0.05, (size, size)).astype(np.float32)

        # Paint a dark disc at center
        cy, cx = 128, 128
        r = 30.0
        yy, xx = np.mgrid[:size, :size]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        image[dist <= r] -= 1.0  # dark disc

        template = make_dark_disc_template(r)
        ncc = _compute_ncc(image, template)

        # Peak should be near (cx, cy)
        peak_y, peak_x = np.unravel_index(np.argmax(ncc), ncc.shape)
        assert abs(peak_x - cx) < 5, f"Peak x={peak_x}, expected ~{cx}"
        assert abs(peak_y - cy) < 5, f"Peak y={peak_y}, expected ~{cy}"

    def test_high_score_at_disc(self):
        """NCC score at disc center should be well above noise."""
        size = 256
        image = np.random.default_rng(42).normal(0, 0.05, (size, size)).astype(np.float32)
        cy, cx = 128, 128
        r = 30.0
        yy, xx = np.mgrid[:size, :size]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        image[dist <= r] -= 1.0

        template = make_dark_disc_template(r)
        ncc = _compute_ncc(image, template)
        assert ncc[cy, cx] > 0.5, f"NCC at disc center = {ncc[cy, cx]:.3f}, expected > 0.5"


# ------------------------------------------------------------------ #
# 3. Multi-scale best radius
# ------------------------------------------------------------------ #

class TestComputeCorrelationMaps:

    def test_best_radius_matches_true(self):
        """Detected radius should be within ±2 steps of the true radius."""
        size = 256
        image = np.random.default_rng(7).normal(0, 0.05, (size, size)).astype(np.float32)
        cx, cy = 128, 128
        true_r = 25.0
        yy, xx = np.mgrid[:size, :size]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        image[dist <= true_r] -= 1.0

        step = 3.0
        radii = np.arange(10.0, 50.0, step)
        best_score, best_radius = compute_correlation_maps(image, radii)

        detected_r = best_radius[cy, cx]
        assert abs(detected_r - true_r) <= 2 * step, (
            f"Detected radius {detected_r:.1f}, true {true_r:.1f}"
        )


# ------------------------------------------------------------------ #
# 4. Peak finding
# ------------------------------------------------------------------ #

class TestFindCorrelationPeaks:

    def test_finds_disc(self):
        """Peak finder should locate a single dark disc."""
        size = 256
        image = np.random.default_rng(99).normal(0, 0.05, (size, size)).astype(np.float32)
        cx, cy = 128, 128
        r = 25.0
        yy, xx = np.mgrid[:size, :size]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        image[dist <= r] -= 1.0

        radii = np.arange(15.0, 40.0, 3.0)
        best_score, best_radius = compute_correlation_maps(image, radii)
        peaks = find_correlation_peaks(best_score, best_radius,
                                       threshold=0.3, min_separation=11, border=50)
        assert peaks.shape[0] >= 1, "No peaks found"

        # Check closest peak to true center
        dists = np.hypot(peaks[:, 0] - cx, peaks[:, 1] - cy)
        assert dists.min() < 10, f"Nearest peak is {dists.min():.1f}px from true center"

    def test_output_shape(self):
        """Peaks array should have shape (N, 5)."""
        score = np.zeros((64, 64), dtype=np.float32)
        radius = np.ones((64, 64), dtype=np.float32) * 10.0
        score[32, 32] = 0.5
        peaks = find_correlation_peaks(score, radius, threshold=0.3)
        assert peaks.shape[1] == 5
        assert peaks.shape[0] >= 1

    def test_sigma_convention(self):
        """sigma = radius / sqrt(2), so diameter = sigma * 2*sqrt(2) = 2*radius."""
        score = np.zeros((64, 64), dtype=np.float32)
        radius_val = 20.0
        radius = np.ones((64, 64), dtype=np.float32) * radius_val
        score[32, 32] = 0.5
        peaks = find_correlation_peaks(score, radius, threshold=0.3)
        sigma = peaks[0, 2]
        expected_sigma = radius_val / np.sqrt(2.0)
        assert abs(sigma - expected_sigma) < 0.1, (
            f"sigma={sigma:.2f}, expected {expected_sigma:.2f}"
        )


# ------------------------------------------------------------------ #
# 5. Integration with conftest fixtures
# ------------------------------------------------------------------ #

def find_match(picks, cx, cy, tolerance_px=40.0):
    """Return the pick closest to (cx, cy), or None if none within tolerance."""
    if picks.shape[0] == 0:
        return None
    dists = np.hypot(picks[:, 0] - cx, picks[:, 1] - cy)
    idx = int(np.argmin(dists))
    if dists[idx] <= tolerance_px:
        return picks[idx]
    return None


def test_template_detects_blobs(small_micrograph):
    """Template matching should detect the 3 Gaussian blobs in the fixture.

    Per-peak border filtering means small-radius picks near edges are kept;
    only peaks whose full template extends past the image are rejected.
    """
    image, blobs = small_micrograph
    # Gaussian blobs don't perfectly match hard-disc templates, use lower threshold
    cfg = _make_cfg(
        dmin=150, dmax=400,
        detection_method="template",
        correlation_threshold=0.05,
        template_radius_step=5.0,
        template_min_separation=11,
    )
    picks = pick_micrograph(image, cfg)
    assert picks.shape[0] > 0, "No picks from template matching"

    for cx, cy, diameter in blobs:
        match = find_match(picks, cx, cy, tolerance_px=50.0)
        assert match is not None, (
            f"Blob at ({cx},{cy}) d={diameter}px not detected by template matching"
        )


def test_combined_mode_detects_blobs(small_micrograph):
    """Combined mode should detect all blobs from both methods."""
    image, blobs = small_micrograph
    cfg = _make_cfg(
        dmin=150, dmax=400,
        detection_method="combined",
        threshold_percentile=99.0,
        correlation_threshold=0.05,
        template_radius_step=5.0,
        template_min_separation=11,
    )
    picks = pick_micrograph(image, cfg)
    assert picks.shape[0] > 0, "No picks from combined mode"

    for cx, cy, diameter in blobs:
        match = find_match(picks, cx, cy, tolerance_px=50.0)
        assert match is not None, (
            f"Blob at ({cx},{cy}) d={diameter}px not detected by combined mode"
        )


# ------------------------------------------------------------------ #
# 6. Empty/noise image
# ------------------------------------------------------------------ #

def test_template_noise_image():
    """Pure noise image should produce few or no template picks."""
    rng = np.random.default_rng(0)
    image = rng.normal(0, 0.01, size=(256, 256)).astype(np.float32)
    cfg = _make_cfg(
        dmin=30, dmax=80,
        detection_method="template",
        correlation_threshold=0.3,
        template_radius_step=5.0,
    )
    picks = pick_micrograph(image, cfg)
    assert picks.shape[0] < 10, f"Too many spurious template picks on noise: {picks.shape[0]}"


# ------------------------------------------------------------------ #
# 7. Config validation
# ------------------------------------------------------------------ #

def test_invalid_detection_method():
    """Invalid detection_method should raise ValueError."""
    with pytest.raises(ValueError, match="detection_method"):
        PickerConfig(detection_method="invalid")


def test_invalid_correlation_threshold():
    """correlation_threshold outside [0, 1] should raise ValueError."""
    with pytest.raises(ValueError, match="correlation_threshold"):
        PickerConfig(detection_method="template", correlation_threshold=1.5)


def test_invalid_template_radius_step():
    """template_radius_step <= 0 should raise ValueError."""
    with pytest.raises(ValueError, match="template_radius_step"):
        PickerConfig(detection_method="template", template_radius_step=-1.0)


def test_dog_k_not_validated_for_template():
    """dog_k validation should be skipped when detection_method='template'."""
    # This should NOT raise even though dog_k default is 1.1 (valid),
    # but explicitly setting dog_k=0.5 should be fine for template-only
    cfg = PickerConfig(detection_method="template", dog_k=0.5)
    assert cfg.dog_k == 0.5
