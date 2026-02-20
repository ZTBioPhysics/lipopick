"""
Multi-resolution image pyramid with anti-aliasing.

Each level downsamples by a given integer factor using:
  1. Gaussian pre-blur (sigma = ds / 2) to prevent aliasing
  2. scipy.ndimage.zoom(1/ds) for resampling

All operations stay in float32.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter, zoom


def build_pyramid_level(
    image: np.ndarray,
    ds: int,
) -> np.ndarray:
    """
    Return a downsampled version of `image` at factor `ds`.

    Parameters
    ----------
    image : ndarray, shape (ny, nx), float32
        Full-resolution micrograph.
    ds : int
        Downsample factor (1 = no downsampling, 2 = half size, 4 = quarter).

    Returns
    -------
    ndarray, float32
        Downsampled image, shape (~ny/ds, ~nx/ds).
    """
    if ds == 1:
        return image.astype(np.float32, copy=False)

    # Anti-aliasing blur: sigma proportional to downsample factor
    sigma = ds / 2.0
    blurred = gaussian_filter(image.astype(np.float32), sigma=sigma)

    # Zoom down (order=1 = bilinear, fast and sufficient)
    factor = 1.0 / ds
    downsampled = zoom(blurred, zoom=factor, order=1, prefilter=False)
    return downsampled.astype(np.float32)


def sigma_range_for_level(
    d_lo: float,
    d_hi: float,
    ds: int,
    k: float,
    min_steps: int = 3,
) -> tuple[float, float, int]:
    """
    Convert full-res diameter range to sigma range at this pyramid level,
    then compute the number of DoG steps needed.

    Parameters
    ----------
    d_lo, d_hi : float
        Full-res diameter range covered by this level (pixels).
    ds : int
        Downsample factor for this level.
    k : float
        Geometric step between sigmas (e.g. 1.10).
    min_steps : int
        Minimum number of sigma steps to compute.

    Returns
    -------
    sigma_min, sigma_max, n_steps : float, float, int
        Sigma values in level-pixels; number of Gaussian blur levels.
    """
    factor = 2.0 * 2.0 ** 0.5   # sigma = diameter / (2√2)
    sigma_min = (d_lo / ds) / factor
    sigma_max = (d_hi / ds) / factor

    # Number of steps: ceil(log(sigma_max/sigma_min) / log(k)) + 1
    n_steps = max(
        min_steps + 1,
        math.ceil(math.log(sigma_max / sigma_min) / math.log(k)) + 1,
    )
    return sigma_min, sigma_max, n_steps
