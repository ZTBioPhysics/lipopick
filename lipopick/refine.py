"""
Optional radial edge refinement.

For each accepted pick, compute a 1-D radial intensity profile averaged
over many angles, then find the edge as the radius with the strongest
absolute gradient.

Two search modes:
  - Default: ±25% of the initial DoG radius estimate (``margin`` parameter).
  - Absolute: [r_min, r_max] bounds regardless of initial estimate.  The
    pipeline always uses absolute mode (r_min=dmin/2, r_max=dmax/2) because
    DoG systematically underestimates hard-edged particle radii.

Uses only numpy/scipy — no scikit-image required.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.ndimage import map_coordinates


def refine_picks(
    image: np.ndarray,
    picks: np.ndarray,
    margin: float = 0.25,
    r_min: float | None = None,
    r_max: float | None = None,
    max_refine_ratio: float = 2.0,
    n_angles: int = 36,
    n_radii: int = 64,
) -> np.ndarray:
    """
    Refine the radius estimate for each pick via radial profiling.

    For each particle, samples a 1-D radial intensity profile averaged over
    many angles, then locates the edge as the radius with the steepest
    gradient (strongest absolute gradient = sharpest edge).

    Parameters
    ----------
    image : ndarray, shape (ny, nx), float32
    picks : ndarray, shape (N, 5), float32
        Columns: x_full, y_full, sigma_full, score, ds
    margin : float
        Search range as fraction of initial radius (default ±25%).
        Ignored when r_min and r_max are both provided.
    r_min, r_max : float or None
        Absolute search bounds (pixels). When both are provided, the radial
        search covers [r_min, r_max] for every pick, regardless of the
        initial sigma estimate.  This is essential when DoG scale detection
        underestimates hard-edged particles.
    max_refine_ratio : float
        Maximum allowed ratio of refined radius to initial DoG radius.
        Prevents small artifacts from being inflated to large diameters
        when the radial search locks onto a distant neighbour's edge.
        Default 2.0 (refined diameter can be at most 2× the initial).
    n_angles : int
        Number of angular samples for radial profile.
    n_radii : int
        Number of radial samples in the profile.

    Returns
    -------
    refined : ndarray, shape (N, 5), float32
        Same format as input with sigma column updated.
    """
    refined = picks.copy()
    ny, nx = image.shape
    angles = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    use_absolute = (r_min is not None and r_max is not None)
    _SIGMA_TO_R = 2.0 * 2.0 ** 0.5 / 2.0   # sigma → radius

    for i, pick in enumerate(picks):
        x0, y0, sigma = float(pick[0]), float(pick[1]), float(pick[2])
        r0 = sigma * _SIGMA_TO_R  # initial DoG radius

        if use_absolute:
            r_lo = max(1.0, r_min)
            r_hi = r_max
        else:
            r_lo = max(1.0, r0 * (1.0 - margin))
            r_hi = r0 * (1.0 + margin)

        # Cap search range by max_refine_ratio relative to initial estimate
        if max_refine_ratio > 0:
            r_hi = min(r_hi, r0 * max_refine_ratio)

        # Clamp r_hi so profile stays inside the image from this pick's center
        max_r = min(x0, y0, nx - 1 - x0, ny - 1 - y0, min(nx, ny) / 2.0)
        r_hi = min(r_hi, max(r_lo + 1, max_r))

        radii = np.linspace(r_lo, r_hi, n_radii)
        profile = np.zeros(n_radii, dtype=np.float32)

        # Average radial profile across all angles
        for r_idx, r in enumerate(radii):
            sample_x = x0 + r * cos_a
            sample_y = y0 + r * sin_a

            sample_x = np.clip(sample_x, 0, nx - 1)
            sample_y = np.clip(sample_y, 0, ny - 1)

            coords = np.array([sample_y, sample_x])
            vals = map_coordinates(image, coords, order=1, mode="nearest")
            profile[r_idx] = vals.mean()

        # Gradient of radial profile.
        # For dark particles on bright background, the edge has the steepest
        # POSITIVE gradient (intensity rises from dark interior to bright bg).
        # For bright particles, the edge has the steepest NEGATIVE gradient.
        # Using absolute value handles both cases automatically.
        grad = np.gradient(profile)
        best_idx = int(np.argmax(np.abs(grad)))

        # If gradient peaks at the search floor (best_idx == 0) when using
        # absolute bounds, the feature's edge is at or below r_min.
        # Keep the original DoG sigma — the detection was valid, refinement
        # just can't improve it because the particle is near the minimum size.
        if use_absolute and best_idx == 0:
            pass  # keep original sigma unchanged
        else:
            r_refined = radii[best_idx]
            sigma_refined = r_refined / _SIGMA_TO_R
            refined[i, 2] = sigma_refined

    return refined
