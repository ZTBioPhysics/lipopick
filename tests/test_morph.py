"""
Tests for morphological cleaning and two-pass detection pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from lipopick.morph import _disk_footprint, morphological_clean


FACTOR = 2.0 * 2.0 ** 0.5  # sigma = diameter / FACTOR


def _make_disk_blob(image, cx, cy, radius, amplitude=-1.0):
    """Add a hard-edged disk blob (not Gaussian) to image in-place."""
    ny, nx = image.shape
    yy, xx = np.ogrid[:ny, :nx]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    image[mask] += amplitude


# ──────────────────────────────────────────────────────────────────────
# Unit tests for morph.py
# ──────────────────────────────────────────────────────────────────────

class TestDiskFootprint:
    """Tests for _disk_footprint."""

    def test_shape(self):
        fp = _disk_footprint(10)
        assert fp.shape == (21, 21)

    def test_center_is_one(self):
        fp = _disk_footprint(5)
        assert fp[5, 5] == 1

    def test_corner_is_zero(self):
        fp = _disk_footprint(10)
        # Corner is at distance sqrt(10^2 + 10^2) = 14.1 > 10
        assert fp[0, 0] == 0

    def test_symmetry(self):
        fp = _disk_footprint(7)
        assert np.array_equal(fp, fp[::-1, :])
        assert np.array_equal(fp, fp[:, ::-1])


class TestMorphologicalClean:
    """Tests for morphological_clean."""

    def test_small_feature_removed(self):
        """A small dark disk (d=60) should be removed by SE_r=50."""
        image = np.zeros((512, 512), dtype=np.float32)
        _make_disk_blob(image, 256, 256, radius=30, amplitude=-1.0)  # d=60

        cleaned = morphological_clean(image, se_radius=50)

        # The small blob should be gone: cleaned interior ≈ 0
        interior = cleaned[226:286, 226:286]  # region around the blob
        assert np.abs(interior).max() < 0.1, \
            f"Small blob not removed: max abs = {np.abs(interior).max():.3f}"

    def test_large_feature_survives(self):
        """A large dark disk (d=200) should survive SE_r=50."""
        image = np.zeros((512, 512), dtype=np.float32)
        _make_disk_blob(image, 256, 256, radius=100, amplitude=-1.0)  # d=200

        cleaned = morphological_clean(image, se_radius=50)

        # The center of the large blob should still be dark
        center_val = cleaned[256, 256]
        assert center_val < -0.5, \
            f"Large blob center should be dark, got {center_val:.3f}"

    def test_output_dtype_and_shape(self):
        """Output should be float32 and same shape as input."""
        image = np.random.default_rng(42).normal(0, 0.1, (256, 256)).astype(np.float32)
        cleaned = morphological_clean(image, se_radius=20)
        assert cleaned.dtype == np.float32
        assert cleaned.shape == image.shape

    def test_no_modification_of_input(self):
        """Input image should not be modified."""
        image = np.random.default_rng(42).normal(0, 0.1, (256, 256)).astype(np.float32)
        original = image.copy()
        morphological_clean(image, se_radius=20)
        assert np.array_equal(image, original)

    def test_smooth_output(self):
        """Closed image at removal site should be smoother than original."""
        image = np.zeros((256, 256), dtype=np.float32)
        _make_disk_blob(image, 128, 128, radius=20, amplitude=-1.0)  # d=40

        cleaned = morphological_clean(image, se_radius=30)  # removes features < 60px

        # Gradient magnitude at the blob location should be smaller after cleaning
        grad_orig = np.hypot(*np.gradient(image[108:148, 108:148]))
        grad_clean = np.hypot(*np.gradient(cleaned[108:148, 108:148]))
        assert grad_clean.max() < grad_orig.max(), \
            f"Cleaned gradient ({grad_clean.max():.4f}) >= original ({grad_orig.max():.4f})"


# ──────────────────────────────────────────────────────────────────────
# Integration tests for two-pass pipeline
# ──────────────────────────────────────────────────────────────────────

class TestTwoPassPipeline:
    """Tests for pass2 pipeline integration.

    Uses 1024×1024 images with well-separated blobs (≥3× diameter apart)
    and appropriate DoG parameters matching the conftest patterns.
    """

    def test_pass2_detects_large_blob(self):
        """Two blobs: small (d=180) + large (d=400) beyond pass-1 dmax.
        Pass-2 should detect the large blob that pass-1 can't reach."""
        from lipopick import PickerConfig, pick_micrograph
        from tests.conftest import make_blob

        rng = np.random.default_rng(123)
        image = rng.normal(0.0, 0.05, (1024, 1024)).astype(np.float32)

        # Small blob at (300, 512), large blob at (750, 512)
        # Separation = 450px > 3 * 180 = 540? No, but > 3 * 60 sigma-wise
        # Both are strong (amplitude -1.0); the test is about SCALE separation.
        make_blob(image, 300, 512, sigma=180 / FACTOR, amplitude=-1.0)
        make_blob(image, 750, 512, sigma=400 / FACTOR, amplitude=-1.0)

        cfg = PickerConfig(
            dmin=150.0, dmax=275.0,
            threshold_percentile=50.0, min_score=0.02,
            nms_beta=0.8, refine=False,
            max_local_contrast=0.0,  # disable for synthetic blobs
            pass2=True,
            closing_radius=80,      # removes features < 160px
            pass2_dmin=250.0, pass2_dmax=500.0,
            pass2_threshold_percentile=50.0,
        )

        picks = pick_micrograph(image, cfg)
        diameters = picks[:, 2] * FACTOR

        # Should have at least 2 picks (1 small from pass-1 + 1 large from pass-2)
        assert picks.shape[0] >= 2, \
            f"Expected >= 2 picks, got {picks.shape[0]}. Diameters: {diameters}"

        # Check that at least one pick is large (d > 250)
        has_large = np.any(diameters > 250)
        assert has_large, \
            f"No large pick detected. Diameters: {diameters}"

    def test_pass2_no_duplicates(self):
        """When both passes detect the same blob, NMS should remove duplicates."""
        from lipopick import PickerConfig, pick_micrograph
        from tests.conftest import make_blob

        rng = np.random.default_rng(456)
        image = rng.normal(0.0, 0.05, (1024, 1024)).astype(np.float32)

        # Single large blob detectable by both passes
        make_blob(image, 512, 512, sigma=200 / FACTOR, amplitude=-1.0)

        cfg = PickerConfig(
            dmin=150.0, dmax=275.0,
            threshold_percentile=50.0, min_score=0.02,
            nms_beta=0.8, refine=False,
            max_local_contrast=0.0,  # disable for synthetic blobs
            pass2=True,
            closing_radius=50,     # small SE — won't remove d=200 blob
            pass2_dmin=100.0, pass2_dmax=300.0,
            pass2_threshold_percentile=50.0,
        )

        picks = pick_micrograph(image, cfg)

        # Should have exactly 1 pick (NMS deduplication)
        assert picks.shape[0] == 1, \
            f"Expected 1 pick after dedup, got {picks.shape[0]}"

    def test_pass2_disabled_by_default(self, small_micrograph):
        """With pass2=False (default), only pass-1 picks are returned."""
        from lipopick import PickerConfig, pick_micrograph

        image, blobs = small_micrograph

        cfg = PickerConfig(
            dmin=150.0, dmax=500.0,
            threshold_percentile=50.0, min_score=0.02,
            nms_beta=0.8, refine=False,
        )
        assert cfg.pass2 is False

        picks = pick_micrograph(image, cfg)
        # small_micrograph has 3 well-separated blobs
        assert picks.shape[0] >= 1, \
            f"Expected >= 1 picks on small_micrograph, got {picks.shape[0]}"

    def test_empty_pass1_skips_pass2(self):
        """If pass-1 finds nothing, pass-2 is not run."""
        from lipopick import PickerConfig, pick_micrograph

        # Pure noise — no blobs to detect
        rng = np.random.default_rng(999)
        image = rng.normal(0.0, 0.01, (512, 512)).astype(np.float32)

        cfg = PickerConfig(
            dmin=150.0, dmax=300.0,
            threshold_percentile=99.9, min_score=1.0,
            nms_beta=0.8, refine=False,
            pass2=True,
            closing_radius=80,
            pass2_dmin=160.0, pass2_dmax=500.0,
        )

        picks = pick_micrograph(image, cfg)
        assert picks.shape[0] == 0


class TestPass2Config:
    """Tests for pass2 config defaults and validation."""

    def test_defaults(self):
        from lipopick import PickerConfig

        cfg = PickerConfig(dmin=50.0, dmax=150.0, pass2=True)
        assert cfg.closing_radius == 50           # int(dmin)
        assert cfg.pass2_dmin == 100.0             # 2 * closing_radius
        assert cfg.pass2_dmax == 300.0             # 2.0 * dmax
        assert cfg.pass2_threshold_percentile == cfg.threshold_percentile

    def test_custom_values(self):
        from lipopick import PickerConfig

        cfg = PickerConfig(
            dmin=50.0, dmax=150.0,
            pass2=True,
            closing_radius=60,
            pass2_dmin=120.0,
            pass2_dmax=400.0,
            pass2_threshold_percentile=50.0,
        )
        assert cfg.closing_radius == 60
        assert cfg.pass2_dmin == 120.0
        assert cfg.pass2_dmax == 400.0
        assert cfg.pass2_threshold_percentile == 50.0

    def test_invalid_pass2_range(self):
        from lipopick import PickerConfig

        with pytest.raises(ValueError, match="pass2_dmin must be less than pass2_dmax"):
            PickerConfig(
                dmin=50.0, dmax=150.0,
                pass2=True,
                pass2_dmin=300.0, pass2_dmax=100.0,
            )
