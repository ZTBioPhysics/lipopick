"""
Tests for lipopick.mask (particle masking for two-pass detection).

1. Feathering smoothness — no sharp edges after masking
2. Local stats matching — masked region has similar mean/std to surroundings
3. No false DoG response — masking a blob doesn't create new DoG peaks there
4. Pass-2 recovers hidden blob — faint blob hidden by strong blob is found
5. Merge deduplication — NMS removes duplicates from overlapping pass-1/pass-2
"""
from __future__ import annotations

import numpy as np
import pytest

from lipopick.mask import mask_particles

# Re-use conftest helpers
from tests.conftest import make_blob, FACTOR


# --------------------------------------------------------------------------- #
# 1. Feathering smoothness
# --------------------------------------------------------------------------- #

def test_feathering_smoothness():
    """Masking a single blob should produce no sharp gradient at the boundary."""
    rng = np.random.default_rng(0)
    image = rng.normal(0.0, 0.05, size=(256, 256)).astype(np.float32)

    # Add a dark blob at center
    sigma = 30.0 / FACTOR
    make_blob(image, 128, 128, sigma, amplitude=-1.0)

    # picks array: (x, y, sigma, score, ds)
    picks = np.array([[128, 128, sigma, 1.0, 1]], dtype=np.float32)

    cleaned = mask_particles(image, picks, feather_width=8.0, dilation=1.1,
                             rng=np.random.default_rng(42))

    # Compute gradient magnitude at the mask boundary region
    # The mask boundary is at r_mask = (sigma * 2√2 / 2) * 1.1
    r = sigma * FACTOR / 2.0
    r_mask = r * 1.1
    r_outer = r_mask + 8.0

    gy, gx = np.gradient(cleaned)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Sample gradient at the transition zone
    yy, xx = np.mgrid[0:256, 0:256]
    dist = np.sqrt((xx - 128)**2 + (yy - 128)**2)
    transition_zone = (dist > r_mask - 2) & (dist < r_outer + 2)

    # The max gradient in the transition zone should be comparable to
    # noise-level gradients, not large discontinuity spikes.
    # For reference, measure gradient in a quiet region far from the blob.
    quiet_zone = dist > r_outer + 20
    if quiet_zone.sum() > 0:
        quiet_grad_p99 = np.percentile(grad_mag[quiet_zone], 99)
        transition_max = np.max(grad_mag[transition_zone])
        # Transition gradients should not be drastically larger than noise.
        # Allow 5x headroom for the cosine ramp.
        assert transition_max < quiet_grad_p99 * 5.0, (
            f"Gradient spike at mask boundary: {transition_max:.4f} vs "
            f"noise 99th pct {quiet_grad_p99:.4f}"
        )


# --------------------------------------------------------------------------- #
# 2. Local stats matching
# --------------------------------------------------------------------------- #

def test_local_stats_matching():
    """Masked region should have similar mean/std to the surrounding background."""
    rng = np.random.default_rng(1)
    bg_mean, bg_std = 0.5, 0.1
    image = rng.normal(bg_mean, bg_std, size=(256, 256)).astype(np.float32)

    sigma = 25.0 / FACTOR
    make_blob(image, 128, 128, sigma, amplitude=-1.0)

    picks = np.array([[128, 128, sigma, 1.0, 1]], dtype=np.float32)
    cleaned = mask_particles(image, picks, feather_width=5.0, dilation=1.1,
                             rng=np.random.default_rng(7))

    # Sample the interior of the masked region
    r = sigma * FACTOR / 2.0
    yy, xx = np.mgrid[0:256, 0:256]
    dist = np.sqrt((xx - 128)**2 + (yy - 128)**2)
    interior = dist < r * 0.8  # well inside mask

    interior_mean = np.mean(cleaned[interior])
    interior_std = np.std(cleaned[interior])

    # Should be close to background stats (within 3 sigma of sampling noise)
    n_interior = interior.sum()
    mean_tolerance = 3 * bg_std / np.sqrt(n_interior)
    assert abs(interior_mean - bg_mean) < mean_tolerance + 0.05, (
        f"Interior mean {interior_mean:.4f} too far from background {bg_mean}"
    )
    # Std should be in the same ballpark
    assert abs(interior_std - bg_std) < bg_std * 0.5, (
        f"Interior std {interior_std:.4f} too different from background {bg_std}"
    )


# --------------------------------------------------------------------------- #
# 3. No false DoG response at masked location
# --------------------------------------------------------------------------- #

def test_dog_response_reduced_after_masking():
    """
    After masking a blob, the DoG response at the blob center should be
    dramatically reduced compared to the original.

    Note: Gaussian test blobs have infinite tails that extend beyond any
    finite mask — this represents the worst case. Real cryo-EM particles
    have hard edges where the mask + dilation fully covers the signal.
    End-to-end pass-2 behavior is validated by test_pass2_recovers_hidden_blob.
    """
    from lipopick.dog import compute_dog_stack
    from lipopick.pyramid import sigma_range_for_level
    from lipopick.config import PickerConfig

    rng = np.random.default_rng(2)
    image = rng.normal(0.0, 0.05, size=(512, 512)).astype(np.float32)

    diameter = 80.0
    sigma = diameter / FACTOR
    make_blob(image, 256, 256, sigma, amplitude=-1.0)

    # Measure original DoG response at blob center
    cfg = PickerConfig(dmin=60, dmax=120)
    ds, d_lo, d_hi = cfg.pyramid_levels[0]
    sigma_min, sigma_max, n_steps = sigma_range_for_level(
        d_lo, d_hi, ds, cfg.dog_k, cfg.dog_min_steps
    )
    dog_original, sigmas = compute_dog_stack(image, sigma_min, n_steps, k=cfg.dog_k)
    original_center = float(np.max(dog_original[:, 256, 256]))

    # Mask the blob
    picks = np.array([[256, 256, sigma, 1.0, 1]], dtype=np.float32)
    cleaned = mask_particles(image, picks, feather_width=5.0, dilation=1.1,
                             rng=np.random.default_rng(99))

    # Measure DoG response at the same location after masking
    dog_cleaned, _ = compute_dog_stack(cleaned, sigma_min, n_steps, k=cfg.dog_k)
    cleaned_center = float(np.max(dog_cleaned[:, 256, 256]))

    # The response at the blob center should be reduced by at least 80%
    assert cleaned_center < original_center * 0.2, (
        f"DoG at masked center not reduced enough: {cleaned_center:.6f} "
        f"vs original {original_center:.6f} "
        f"(reduction: {1 - cleaned_center/original_center:.0%})"
    )


# --------------------------------------------------------------------------- #
# 4. Pass-2 recovers hidden blob
# --------------------------------------------------------------------------- #

def test_pass2_recovers_hidden_blob():
    """
    Two-pass detection should recover a faint large blob that is missed
    when a strong small blob is nearby.

    Uses a larger image (1024) so blobs are well separated and don't
    interfere with edge exclusion or each other's DoG response.
    """
    from lipopick.config import PickerConfig
    from lipopick.pipeline import pick_micrograph

    rng = np.random.default_rng(10)
    image = rng.normal(0.0, 0.02, size=(1024, 1024)).astype(np.float32)

    # Strong small blob at one location
    small_d = 80.0
    small_sigma = small_d / FACTOR
    make_blob(image, 300, 512, small_sigma, amplitude=-1.0)

    # Moderate large blob far away — amplitude strong enough for DoG to
    # detect on the cleaned image, but in a different size range.
    large_d = 200.0
    large_sigma = large_d / FACTOR
    make_blob(image, 700, 512, large_sigma, amplitude=-0.5)

    # Pass 1 only: should find the small blob; may or may not find
    # the large one (its diameter is outside [50, 120])
    cfg1 = PickerConfig(
        dmin=50, dmax=120,
        threshold_percentile=50.0, min_score=0.005,
        nms_beta=1.0,
        max_local_contrast=0,  # disable for synthetic data
        pass2=False,
    )
    picks1 = pick_micrograph(image, cfg1)
    assert picks1.shape[0] >= 1, "Pass 1 should find at least the small blob"

    # Two-pass: should find both
    cfg2 = PickerConfig(
        dmin=50, dmax=120,
        threshold_percentile=50.0, min_score=0.005,
        nms_beta=1.0,
        max_local_contrast=0,  # disable for synthetic data
        pass2=True,
        pass2_dmin=120,
        pass2_dmax=300,
        pass2_threshold_percentile=50.0,
        mask_feather_width=5.0,
        mask_dilation=1.1,
    )
    picks2 = pick_micrograph(image, cfg2)

    # Should have at least 2 picks (small + large)
    assert picks2.shape[0] >= 2, (
        f"Two-pass should find >=2 particles, got {picks2.shape[0]}"
    )

    # Check there's a pick near the large blob location
    dist_to_large = np.sqrt(
        (picks2[:, 0] - 700)**2 + (picks2[:, 1] - 512)**2
    )
    assert np.any(dist_to_large < large_d), (
        f"No pass-2 pick found near the large blob "
        f"(closest: {dist_to_large.min():.1f}px)"
    )


# --------------------------------------------------------------------------- #
# 5. Merge deduplication
# --------------------------------------------------------------------------- #

def test_merge_deduplication():
    """
    When pass-1 and pass-2 size ranges overlap, NMS should deduplicate
    picks at the same location.
    """
    from lipopick.config import PickerConfig
    from lipopick.pipeline import pick_micrograph

    rng = np.random.default_rng(20)
    image = rng.normal(0.0, 0.02, size=(1024, 1024)).astype(np.float32)

    # Single strong blob that falls in both pass-1 and pass-2 ranges
    diameter = 120.0
    sigma = diameter / FACTOR
    make_blob(image, 512, 512, sigma, amplitude=-1.0)

    cfg = PickerConfig(
        dmin=80, dmax=160,
        threshold_percentile=50.0, min_score=0.005,
        nms_beta=1.0,
        max_local_contrast=0,  # disable for synthetic data
        pass2=True,
        pass2_dmin=100,  # overlaps with pass-1 range
        pass2_dmax=250,
        pass2_threshold_percentile=50.0,
    )
    picks = pick_micrograph(image, cfg)

    # Should have at least 1 pick
    assert picks.shape[0] >= 1, f"Expected at least 1 pick, got {picks.shape[0]}"

    # Count picks near the blob center — should be exactly 1 after NMS
    dist_to_blob = np.sqrt((picks[:, 0] - 512)**2 + (picks[:, 1] - 512)**2)
    close = dist_to_blob < diameter
    assert close.sum() == 1, (
        f"Expected 1 pick at blob, got {close.sum()} (NMS deduplication failed)"
    )


# --------------------------------------------------------------------------- #
# 6. mask_particles returns copy (original unchanged)
# --------------------------------------------------------------------------- #

def test_mask_does_not_modify_original():
    """mask_particles should return a new array, not modify the input."""
    rng = np.random.default_rng(3)
    image = rng.normal(0.0, 0.1, size=(128, 128)).astype(np.float32)
    original = image.copy()

    sigma = 15.0 / FACTOR
    make_blob(image, 64, 64, sigma, amplitude=-1.0)
    original_with_blob = image.copy()

    picks = np.array([[64, 64, sigma, 1.0, 1]], dtype=np.float32)
    cleaned = mask_particles(image, picks, rng=np.random.default_rng(0))

    # Original should be unchanged
    np.testing.assert_array_equal(image, original_with_blob)
    # Cleaned should differ (the blob was masked)
    assert not np.array_equal(cleaned, image)


# --------------------------------------------------------------------------- #
# 7. Empty picks → identity
# --------------------------------------------------------------------------- #

def test_mask_empty_picks():
    """Masking with zero picks should return an identical copy."""
    rng = np.random.default_rng(4)
    image = rng.normal(0.0, 0.1, size=(64, 64)).astype(np.float32)

    picks = np.empty((0, 5), dtype=np.float32)
    cleaned = mask_particles(image, picks)

    np.testing.assert_array_equal(cleaned, image)
    assert cleaned is not image  # should be a copy
