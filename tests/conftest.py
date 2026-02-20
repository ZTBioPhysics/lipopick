"""
Shared fixtures for lipopick tests.

Provides synthetic micrographs with known Gaussian blobs.

Cryo-EM convention: particles appear DARK (negative amplitude) on a lighter background.
DoG = G(bigger) - G(smaller) gives POSITIVE response at dark blob centers.

Blob placement rules:
  - Separation ≥ 3× blob diameter to avoid DoG cross-contamination at large scales
  - Blobs at least dmax/2 from image edges (to avoid edge exclusion)
  - Image size ≥ 4 × dmax so blobs are comfortably separated
"""
from __future__ import annotations

import numpy as np
import pytest


FACTOR = 2.0 * 2.0 ** 0.5   # sigma = diameter / FACTOR


def make_blob(
    image: np.ndarray,
    cx: int,
    cy: int,
    sigma: float,
    amplitude: float = -1.0,
) -> None:
    """
    Add an isotropic Gaussian blob to `image` in-place.

    Default amplitude is NEGATIVE to match cryo-EM convention:
    particles appear dark (high electron scattering) on a lighter background.
    DoG = G(bigger) - G(smaller) gives positive response at dark blob centers.
    """
    ny, nx = image.shape
    ys = np.arange(ny)[:, None]
    xs = np.arange(nx)[None, :]
    image += amplitude * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))


@pytest.fixture
def small_micrograph():
    """
    1024×1024 micrograph with three well-separated blobs at different sizes.
    Blobs are spaced >3× their diameter apart to avoid DoG cross-contamination.
    Returns (image, blobs) where blobs = list of (cx, cy, diameter_px).
    """
    rng = np.random.default_rng(42)
    image = rng.normal(0.0, 0.05, size=(1024, 1024)).astype(np.float32)

    blobs = [
        (200, 200, 180.0),   # small — fits in ds=1 level (150–275px)
        (750, 200, 250.0),   # medium — fits in ds=1/ds=2 overlap
        (500, 750, 320.0),   # large — fits in ds=2 level (225–440px)
    ]
    # Separations: B1–B2=550px (3.1×d=180), B1–B3=628px, B2–B3=604px
    for cx, cy, d in blobs:
        sigma = d / FACTOR
        make_blob(image, cx, cy, sigma, amplitude=-1.0)

    return image, blobs


@pytest.fixture
def large_micrograph():
    """
    1024×1024 micrograph with blobs at three different scale ranges.
    Returns (image, blobs) where blobs = list of (cx, cy, diameter_px).
    """
    rng = np.random.default_rng(7)
    image = rng.normal(0.0, 0.05, size=(1024, 1024)).astype(np.float32)

    blobs = [
        (200, 200, 180.0),    # small — ds=1 level (150–275px)
        (750, 200, 350.0),    # medium — ds=2 level (225–440px)
        (450, 750, 460.0),    # large — ds=4 level (360–500px)
    ]
    # Separations: B1–B2=550px (3.1×d=180), B1–B3=676px, B2–B3=611px
    for cx, cy, d in blobs:
        sigma = d / FACTOR
        make_blob(image, cx, cy, sigma, amplitude=-1.0)

    return image, blobs


@pytest.fixture
def packed_micrograph():
    """
    512×512 micrograph with two same-sized blobs far enough apart to both survive NMS.
    Returns (image, blobs).
    """
    rng = np.random.default_rng(99)
    image = rng.normal(0.0, 0.05, size=(512, 512)).astype(np.float32)

    blobs = [
        (100, 256, 160.0),   # left — sigma=56.6, r=80px, exclusion=64px
        (400, 256, 160.0),   # right — separation=300px >> 2*64=128px
    ]
    for cx, cy, d in blobs:
        sigma = d / FACTOR
        make_blob(image, cx, cy, sigma, amplitude=-1.0)

    return image, blobs
