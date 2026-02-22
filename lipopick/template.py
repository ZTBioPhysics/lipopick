"""
Dark-disc template matching detector for cryo-EM lipoprotein micrographs.

Complements the DoG detector for large, faint (lipid-rich) particles that DoG
cannot detect.  Uses FFT-based normalized cross-correlation (NCC) with a
dark-disc + bright-annulus template at multiple radii.

Key advantage: NCC integrates signal over the full disc area, so even faint
particles accumulate detectable correlation.  The bright annulus naturally
distinguishes real particles from clusters of small particles (which have
bright gaps in the interior).
"""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


def make_dark_disc_template(
    radius: float,
    annulus_width_frac: float = 0.5,
    max_annulus_width: float = 40.0,
) -> np.ndarray:
    """
    Build a zero-mean dark-disc + bright-annulus template.

    Parameters
    ----------
    radius : float
        Inner disc radius in pixels.
    annulus_width_frac : float
        Annulus width as a fraction of radius.
    max_annulus_width : float
        Cap on annulus width in pixels.

    Returns
    -------
    template : ndarray, float32
        Square array of shape (2*R_out+1, 2*R_out+1), zero-mean.
        Inner disc pixels = -1.0, annulus pixels = +n_inner/n_annulus.
    """
    annulus_width = min(annulus_width_frac * radius, max_annulus_width)
    r_out = radius + annulus_width
    size = int(np.ceil(r_out)) * 2 + 1
    center = size // 2

    yy, xx = np.mgrid[:size, :size]
    dist = np.sqrt((xx - center) ** 2 + (yy - center) ** 2)

    template = np.zeros((size, size), dtype=np.float32)
    inner_mask = dist <= radius
    annulus_mask = (dist > radius) & (dist <= r_out)

    n_inner = int(inner_mask.sum())
    n_annulus = int(annulus_mask.sum())

    if n_inner == 0 or n_annulus == 0:
        return template

    # Set values so template is zero-mean:
    # inner * n_inner + annulus * n_annulus = 0
    # inner = -1.0  =>  annulus = n_inner / n_annulus
    template[inner_mask] = -1.0
    template[annulus_mask] = float(n_inner) / float(n_annulus)

    return template


def _compute_ncc(
    image: np.ndarray,
    template: np.ndarray,
) -> np.ndarray:
    """
    FFT-based normalized cross-correlation.

    The template must be zero-mean.  NCC is computed as::

        ncc = CC / (n * local_std * template_std)

    where CC = fftconvolve(image, template) already accounts for the
    zero-mean template (subtracting the local mean implicitly).

    Parameters
    ----------
    image : ndarray, 2D float32
    template : ndarray, 2D float32, zero-mean

    Returns
    -------
    ncc : ndarray, same shape as image, float32, clipped to [-1, 1]
    """
    template_std = float(np.std(template))
    if template_std < 1e-12:
        return np.zeros_like(image, dtype=np.float32)

    n = float(template.size)

    # Cross-correlation (template is symmetric, no flip needed)
    cc = fftconvolve(image, template, mode="same")

    # Local statistics under the template footprint
    ones = np.ones_like(template)
    local_sum = fftconvolve(image, ones, mode="same")
    local_sum_sq = fftconvolve(image ** 2, ones, mode="same")

    local_var = local_sum_sq / n - (local_sum / n) ** 2
    local_std = np.sqrt(np.clip(local_var, 1e-12, None))

    ncc = cc / (n * local_std * template_std)
    ncc = np.clip(ncc, -1.0, 1.0).astype(np.float32)

    return ncc


def compute_correlation_maps(
    image: np.ndarray,
    radii: np.ndarray,
    annulus_width_frac: float = 0.5,
    max_annulus_width: float = 40.0,
) -> tuple:
    """
    Compute best NCC score and corresponding radius at each pixel.

    Memory-efficient: processes one template at a time and tracks only
    the best score and radius maps.

    Parameters
    ----------
    image : ndarray, 2D float32
    radii : ndarray, 1D
        Template radii to evaluate (pixels).
    annulus_width_frac : float
    max_annulus_width : float

    Returns
    -------
    best_score : ndarray, same shape as image, float32
        Maximum NCC across all radii at each pixel.
    best_radius : ndarray, same shape as image, float32
        Radius that achieved the best NCC at each pixel.
    """
    best_score = np.full(image.shape, -np.inf, dtype=np.float32)
    best_radius = np.zeros(image.shape, dtype=np.float32)

    for r in radii:
        template = make_dark_disc_template(
            r, annulus_width_frac, max_annulus_width,
        )
        ncc = _compute_ncc(image, template)

        improved = ncc > best_score
        best_score[improved] = ncc[improved]
        best_radius[improved] = float(r)

    # Replace -inf with 0 where no template was evaluated
    best_score[best_score == -np.inf] = 0.0

    return best_score, best_radius


def find_correlation_peaks(
    best_score: np.ndarray,
    best_radius: np.ndarray,
    threshold: float = 0.15,
    min_separation: int = 5,
    border: int = 0,
) -> np.ndarray:
    """
    Extract local maxima from the best-score map.

    Parameters
    ----------
    best_score : ndarray, 2D float32
    best_radius : ndarray, 2D float32
    threshold : float
        Minimum absolute NCC score to accept.
    min_separation : int
        Neighborhood size for ``maximum_filter`` (local maxima detection).
    border : int
        Exclude peaks within this many pixels of the image edge.

    Returns
    -------
    candidates : ndarray, shape (N, 5), float32
        Columns: x, y, sigma, score, ds.
        sigma = radius / sqrt(2), maintaining diameter = sigma * 2*sqrt(2).
        ds = 1 (no pyramid for template matching).
    """
    from scipy.ndimage import maximum_filter

    ny, nx = best_score.shape

    # Ensure min_separation is odd for symmetric neighborhood
    if min_separation % 2 == 0:
        min_separation += 1

    local_max = maximum_filter(best_score, size=min_separation)
    peak_mask = (best_score == local_max) & (best_score >= threshold)

    # Border exclusion
    if border > 0:
        peak_mask[:border, :] = False
        peak_mask[-border:, :] = False
        peak_mask[:, :border] = False
        peak_mask[:, -border:] = False

    ys, xs = np.where(peak_mask)
    if len(xs) == 0:
        return np.empty((0, 5), dtype=np.float32)

    scores = best_score[ys, xs]
    radii = best_radius[ys, xs]

    # Convert radius to sigma: diameter = 2*radius, sigma = diameter / (2*sqrt(2))
    # => sigma = radius / sqrt(2)
    sigmas = radii / np.sqrt(2.0)

    candidates = np.column_stack([
        xs.astype(np.float32),
        ys.astype(np.float32),
        sigmas.astype(np.float32),
        scores.astype(np.float32),
        np.ones(len(xs), dtype=np.float32),  # ds = 1
    ])

    return candidates
