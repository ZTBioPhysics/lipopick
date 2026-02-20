"""Tests for dog.py — DoG stack computation and local maxima detection."""
import numpy as np
import pytest

from lipopick.dog import compute_dog_stack, find_local_maxima
from tests.conftest import make_blob


def make_single_blob_image(size=256, cx=128, cy=128, diameter=180.0):
    factor = 2.0 * 2.0 ** 0.5
    sigma = diameter / factor
    img = np.zeros((size, size), dtype=np.float32)
    make_blob(img, cx, cy, sigma, amplitude=-1.0)
    return img, sigma


def test_dog_stack_shape():
    img, sigma_blob = make_single_blob_image()
    sigma_min = sigma_blob * 0.7
    n_steps = 8
    stack, sigmas = compute_dog_stack(img, sigma_min=sigma_min, n_steps=n_steps, k=1.10)
    assert stack.shape == (n_steps - 1, img.shape[0], img.shape[1])
    assert stack.dtype == np.float32
    assert len(sigmas) == n_steps - 1


def test_dog_stack_has_positive_responses():
    """DoG of a positive blob should have at least some positive values."""
    img, sigma_blob = make_single_blob_image()
    sigma_min = sigma_blob * 0.7
    stack, _ = compute_dog_stack(img, sigma_min=sigma_min, n_steps=8, k=1.10)
    assert stack.max() > 0


def test_find_local_maxima_detects_blob():
    """find_local_maxima should find a candidate near the known blob center."""
    cx, cy, diameter = 128, 128, 180.0
    img, sigma_blob = make_single_blob_image(cx=cx, cy=cy, diameter=diameter)
    sigma_min = sigma_blob * 0.6
    n_steps = 10
    stack, sigmas = compute_dog_stack(img, sigma_min=sigma_min, n_steps=n_steps, k=1.10)
    candidates = find_local_maxima(stack, sigmas, threshold_percentile=90.0, ds=1)

    assert candidates.shape[0] > 0, "No candidates found"

    # Best candidate should be close to (cx, cy)
    best = candidates[np.argmax(candidates[:, 3])]   # column 3 = score
    x_found, y_found = best[0], best[1]
    dist = np.hypot(x_found - cx, y_found - cy)
    assert dist < 20.0, f"Peak too far from blob center: dist={dist:.1f}px"


def test_find_local_maxima_diameter_accuracy():
    """Recovered diameter should be within 20% of true diameter."""
    cx, cy, diameter = 128, 128, 200.0
    factor = 2.0 * 2.0 ** 0.5
    true_sigma = diameter / factor
    img, _ = make_single_blob_image(cx=cx, cy=cy, diameter=diameter)
    sigma_min = true_sigma * 0.5
    n_steps = 15
    stack, sigmas = compute_dog_stack(img, sigma_min=sigma_min, n_steps=n_steps, k=1.10)
    candidates = find_local_maxima(stack, sigmas, threshold_percentile=90.0, ds=1)

    assert candidates.shape[0] > 0

    best = candidates[np.argmax(candidates[:, 3])]
    recovered_diameter = best[2] * factor   # sigma → diameter
    error_frac = abs(recovered_diameter - diameter) / diameter
    assert error_frac < 0.20, f"Diameter error too large: {error_frac*100:.1f}%"
