"""
Particle masking for two-pass detection (mask-and-redetect).

Replaces detected particles with locally-matched Gaussian noise using
cosine-feathered blending to avoid sharp mask boundaries that would
create false DoG responses.
"""
from __future__ import annotations

import numpy as np


# Column indices matching pipeline.py convention
_X = 0
_Y = 1
_SIGMA = 2
_SCORE = 3

_SIGMA_TO_DIAM = 2.0 * 2.0 ** 0.5   # diameter = sigma * 2√2


def mask_particles(
    image: np.ndarray,
    picks: np.ndarray,
    feather_width: float = 5.0,
    dilation: float = 1.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Replace detected particles with locally-matched Gaussian noise.

    Each pick is masked with a cosine-feathered blend: the interior is
    fully replaced with noise matching the local background statistics,
    and a smooth transition zone prevents sharp edges that DoG would
    detect as false particles.

    Parameters
    ----------
    image : ndarray, shape (ny, nx), float32
        Original micrograph (not modified).
    picks : ndarray, shape (N, 5), float32
        Columns: x, y, sigma, score, ds.
    feather_width : float
        Width of cosine transition zone in pixels.
    dilation : float
        Factor to expand mask radius beyond particle radius (1.1 = 10%).
        Ensures the full DoG response zone is masked.
    rng : numpy.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    cleaned : ndarray, shape (ny, nx), float32
        Copy of image with particles replaced by noise.
    """
    if picks.shape[0] == 0:
        return image.copy()

    if rng is None:
        rng = np.random.default_rng()

    ny, nx = image.shape
    cleaned = image.copy()

    # Process picks in descending score order (strongest first) so
    # overlapping masks use the most reliable local statistics.
    order = np.argsort(picks[:, _SCORE])[::-1]

    for idx in order:
        cx = float(picks[idx, _X])
        cy = float(picks[idx, _Y])
        sigma = float(picks[idx, _SIGMA])

        # Particle radius from sigma, then dilate
        r = sigma * _SIGMA_TO_DIAM / 2.0
        r_mask = r * dilation
        r_outer = r_mask + feather_width

        # Bounding box for the affected region
        x0 = max(0, int(cx - r_outer) - 1)
        x1 = min(nx, int(cx + r_outer) + 2)
        y0 = max(0, int(cy - r_outer) - 1)
        y1 = min(ny, int(cy + r_outer) + 2)

        if x1 <= x0 or y1 <= y0:
            continue

        # Distance map for this patch
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(np.float32)

        # Compute local background stats from annulus (1.5r to 2.5r)
        r_ann_inner = 1.5 * r
        r_ann_outer = 2.5 * r
        annulus_mask = (dist >= r_ann_inner) & (dist <= r_ann_outer)

        if annulus_mask.sum() > 0:
            annulus_pixels = cleaned[y0:y1, x0:x1][annulus_mask]
            local_mean = float(np.mean(annulus_pixels))
            local_std = float(np.std(annulus_pixels))
        else:
            # Fallback to global stats if annulus is outside image
            local_mean = float(np.mean(cleaned))
            local_std = float(np.std(cleaned))

        # Ensure std is not zero (uniform image edge case)
        if local_std < 1e-9:
            local_std = float(np.std(cleaned))
        if local_std < 1e-9:
            local_std = 1.0

        # Generate replacement noise
        noise = rng.normal(local_mean, local_std, size=(y1 - y0, x1 - x0)).astype(np.float32)

        # Cosine-feathered blending weight
        #   dist <= r_mask:                    weight = 1.0 (full replacement)
        #   r_mask < dist <= r_mask + feather: weight = 0.5*(1 + cos(...))
        #   dist > r_mask + feather:           weight = 0.0 (original)
        weight = np.zeros_like(dist)
        inner = dist <= r_mask
        weight[inner] = 1.0

        transition = (dist > r_mask) & (dist <= r_outer)
        if transition.any():
            t = (dist[transition] - r_mask) / feather_width
            weight[transition] = 0.5 * (1.0 + np.cos(np.pi * t))

        # Blend: cleaned = weight * noise + (1 - weight) * original
        patch = cleaned[y0:y1, x0:x1]
        cleaned[y0:y1, x0:x1] = weight * noise + (1.0 - weight) * patch

    return cleaned
