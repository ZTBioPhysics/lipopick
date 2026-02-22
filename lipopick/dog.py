"""
DoG (Difference-of-Gaussians) scale-space detection.

For each pyramid level:
  1. Build a geometric sigma series: sigma_i = sigma_min * k^i
  2. Compute DoG_i = G(sigma_i * k) - G(sigma_i)  (raw, no sigma² normalization)
  3. Stack DoGs into a 3-D volume (n_scales, ny, nx)
  4. Find 3-D local maxima

The raw DoG (without sigma² normalization) correctly identifies the matched scale:
for a disk-shaped blob of radius R, the raw DoG at the center peaks at sigma ≈ R/sqrt(2),
so diameter = sigma * 2√2. The sigma² normalization shifts the optimum to sigma_max,
biasing diameter estimates toward the upper end of the search range.

Memory-efficient: only two Gaussian images (g_prev, g_curr) live in memory
at once; the DoG is computed on the fly and accumulated in the stack.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter


def compute_dog_stack(
    image: np.ndarray,
    sigma_min: float,
    n_steps: int,
    k: float = 1.10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the raw DoG stack for one pyramid level.

    Parameters
    ----------
    image : ndarray, shape (ny, nx), float32
    sigma_min : float
        Smallest sigma in the series (level-pixels).
    n_steps : int
        Number of Gaussian blur levels (produces n_steps-1 DoG layers).
    k : float
        Geometric ratio between consecutive sigmas.

    Returns
    -------
    dog_stack : ndarray, shape (n_steps-1, ny, nx), float32
        Raw DoG responses (G(k*sigma) - G(sigma)), no sigma² normalization.
    sigmas : ndarray, shape (n_steps-1,), float32
        The *lower* sigma for each DoG layer (useful for diameter recovery).
    """
    ny, nx = image.shape
    n_dogs = n_steps - 1
    dog_stack = np.empty((n_dogs, ny, nx), dtype=np.float32)
    sigmas = np.empty(n_dogs, dtype=np.float32)

    # mode='nearest' avoids reflection artifacts at image borders that can inflate
    # the threshold and mask real blobs when sigma is large relative to image size.
    g_prev = gaussian_filter(image, sigma=sigma_min, mode='nearest').astype(np.float32)
    sigma_curr = sigma_min

    for i in range(n_dogs):
        sigma_next = sigma_curr * k
        g_curr = gaussian_filter(image, sigma=sigma_next, mode='nearest').astype(np.float32)

        # Raw DoG (no sigma² normalization — see module docstring for rationale)
        dog_stack[i] = g_curr - g_prev

        sigmas[i] = sigma_curr
        g_prev = g_curr
        sigma_curr = sigma_next

    return dog_stack, sigmas


def find_local_maxima(
    dog_stack: np.ndarray,
    sigmas: np.ndarray,
    threshold_percentile: float = 99.7,
    min_score: float = 0.0,
    ds: int = 1,
    border: int = 0,
) -> np.ndarray:
    """
    Find 3-D local maxima in the DoG scale-space volume.

    Parameters
    ----------
    dog_stack : ndarray, shape (n_dogs, ny, nx), float32
    sigmas : ndarray, shape (n_dogs,), float32
        Lower sigma for each DoG layer.
    threshold_percentile : float
        Percentile of the *positive* DoG values used as threshold.
    min_score : float
        Absolute minimum score (applied after percentile threshold).
    ds : int
        Downsample factor — used to map coordinates to full-res.
    border : int
        Exclude this many border pixels from threshold computation.

    Returns
    -------
    candidates : ndarray, shape (N, 5), float32
        Columns: x_full, y_full, sigma_full, score, ds
        All coordinates in *full-resolution* pixels.
    """
    # For cryo-EM: particles are dark (high electron scattering) on a lighter background.
    # G(bigger) - G(smaller) gives POSITIVE response at dark blob centers.
    # We find local maxima of the positive DoG.

    # 3-D local maxima: a voxel is a peak if it equals the max in a 3x3x3 neighbourhood
    local_max = maximum_filter(dog_stack, size=(3, 3, 3), mode="constant", cval=0.0)
    is_peak = (dog_stack == local_max) & (dog_stack > 0)

    # Threshold: compute percentile on interior pixels to avoid edge boundary artifacts
    # (mode='nearest' in Gaussian filter largely prevents these, but this is belt-and-suspenders)
    if border > 0 and dog_stack.shape[1] > 2 * border and dog_stack.shape[2] > 2 * border:
        interior = dog_stack[:, border:-border, border:-border]
    else:
        interior = dog_stack
    pos_vals = interior[interior > 0]
    if pos_vals.size == 0:
        return np.empty((0, 5), dtype=np.float32)

    threshold = max(
        float(np.percentile(pos_vals, threshold_percentile)),
        min_score,
    )

    is_peak &= (dog_stack >= threshold)

    # Extract peak indices
    scale_idx, y_idx, x_idx = np.where(is_peak)
    if len(scale_idx) == 0:
        return np.empty((0, 5), dtype=np.float32)

    scores = dog_stack[scale_idx, y_idx, x_idx].astype(np.float32)
    sigma_vals = sigmas[scale_idx].astype(np.float32)

    # Map to full-resolution coordinates
    x_full = (x_idx * ds).astype(np.float32)
    y_full = (y_idx * ds).astype(np.float32)
    sigma_full = (sigma_vals * ds).astype(np.float32)
    ds_col = np.full(len(scores), ds, dtype=np.float32)

    candidates = np.column_stack([x_full, y_full, sigma_full, scores, ds_col])
    return candidates
