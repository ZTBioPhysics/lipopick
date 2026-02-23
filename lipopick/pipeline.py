"""
Orchestrator: ties all stages together into pick_micrograph() and process_batch().

Stages per micrograph (pick_micrograph):
  1. Detection (DoG, template matching, or combined)
  2. Edge exclusion (reject picks within dmax/2 of border)
  3. Size-aware NMS
  4. Optional radial refinement
  5. Post-refine diameter filter (clip to [dmin, dmax])
  6. Pixel-statistics filter (bright interior / ambiguous / contaminant)
  7. Post-refinement overlap filter (greedy, score-ordered)
"""
from __future__ import annotations

import multiprocessing
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import PickerConfig
from .dog import compute_dog_stack, find_local_maxima
from .extraction import make_extraction_plan
from .io import (
    read_micrograph,
    write_picks_csv,
    write_picks_star,
    write_extraction_plan,
    write_combined_csv,
    list_micrographs,
    PICKS_CSV_FIELDS,
)
from .nms import size_aware_nms
from scipy.spatial import cKDTree
from .pyramid import build_pyramid_level, sigma_range_for_level
from .viz import plot_picks_overlay, plot_size_histogram, plot_batch_summary


# Column indices in candidate / picks arrays
_X = 0
_Y = 1
_SIGMA = 2
_SCORE = 3
_DS = 4

_SIGMA_TO_DIAM = 2.0 * 2.0 ** 0.5   # diameter = sigma * SIGMA_TO_DIAM


def _detect_dog(
    image: np.ndarray,
    cfg: PickerConfig,
) -> np.ndarray:
    """
    DoG detection: build pyramid, compute scale-space, extract local maxima.

    Returns
    -------
    candidates : ndarray, shape (N, 5), float32
    """
    all_candidates: List[np.ndarray] = []

    for ds, d_lo, d_hi in cfg.pyramid_levels:
        level_img = build_pyramid_level(image, ds)

        sigma_min, sigma_max, n_steps = sigma_range_for_level(
            d_lo, d_hi, ds, cfg.dog_k, cfg.dog_min_steps
        )

        dog_stack, sigmas = compute_dog_stack(level_img, sigma_min, n_steps, k=cfg.dog_k)
        sigma_max_level = float(sigmas[-1]) * cfg.dog_k
        border_level = min(
            int(np.ceil(sigma_max_level * 2)),
            min(level_img.shape[0], level_img.shape[1]) // 4,
        )
        candidates = find_local_maxima(
            dog_stack, sigmas,
            threshold_percentile=cfg.threshold_percentile,
            min_score=cfg.min_score,
            ds=ds,
            border=border_level,
        )
        all_candidates.append(candidates)
        del dog_stack

    if not all_candidates or all(c.shape[0] == 0 for c in all_candidates):
        return np.empty((0, 5), dtype=np.float32)

    return np.vstack([c for c in all_candidates if c.shape[0] > 0])


def _detect_template(
    image: np.ndarray,
    cfg: PickerConfig,
) -> np.ndarray:
    """
    Template matching detection: dark-disc NCC at multiple radii.

    Returns
    -------
    candidates : ndarray, shape (N, 5), float32
    """
    from .template import compute_correlation_maps, find_correlation_peaks

    r_min = cfg.dmin / 2.0
    r_max = cfg.dmax / 2.0
    radii = np.arange(r_min, r_max + cfg.template_radius_step, cfg.template_radius_step)

    best_score, best_radius = compute_correlation_maps(
        image, radii, cfg.annulus_width_fraction, cfg.max_annulus_width,
    )
    candidates = find_correlation_peaks(
        best_score, best_radius,
        threshold=cfg.correlation_threshold,
        min_separation=cfg.template_min_separation,
        border=int(np.ceil(r_min)),  # minimal border; per-peak filter below
    )

    # Per-peak border filter: reject peaks whose detected template
    # (disc + annulus) extends past the image edge.  This is more
    # permissive than a global border based on r_max, so small-radius
    # picks near edges are kept.
    if candidates.shape[0] > 0:
        ny, nx = image.shape
        peak_r = candidates[:, _SIGMA] * _SIGMA_TO_DIAM / 2.0  # detected radius
        annulus_w = np.minimum(
            cfg.annulus_width_fraction * peak_r, cfg.max_annulus_width,
        )
        r_out = peak_r + annulus_w
        keep = (
            (candidates[:, _X] >= r_out) &
            (candidates[:, _X] <= nx - 1 - r_out) &
            (candidates[:, _Y] >= r_out) &
            (candidates[:, _Y] <= ny - 1 - r_out)
        )
        candidates = candidates[keep]

    return candidates


def pick_micrograph(
    image: np.ndarray,
    cfg: PickerConfig,
) -> np.ndarray:
    """
    Run the full detection pipeline on a single micrograph array.

    Parameters
    ----------
    image : ndarray, shape (ny, nx), float32
    cfg : PickerConfig

    Returns
    -------
    picks : ndarray, shape (N, 5), float32
        Columns: x_full, y_full, sigma_full, score, ds
    """
    ny, nx = image.shape

    # ── Detection phase ───────────────────────────────────────────────
    if cfg.detection_method == "dog":
        candidates = _detect_dog(image, cfg)
    elif cfg.detection_method == "template":
        candidates = _detect_template(image, cfg)
    elif cfg.detection_method == "combined":
        dog_cands = _detect_dog(image, cfg)
        tmpl_cands = _detect_template(image, cfg)
        parts = [c for c in (dog_cands, tmpl_cands) if c.shape[0] > 0]
        candidates = np.vstack(parts) if parts else np.empty((0, 5), dtype=np.float32)
    else:
        raise ValueError(f"Unknown detection_method: {cfg.detection_method!r}")

    if candidates.shape[0] == 0:
        return np.empty((0, 5), dtype=np.float32)

    # ── Post-processing (identical for all methods) ───────────────────

    # Edge exclusion: reject within dmax/2 of image border
    border = cfg.dmax / 2.0 * cfg.edge_fraction
    edge_mask = (
        (candidates[:, _X] < border) |
        (candidates[:, _X] > nx - 1 - border) |
        (candidates[:, _Y] < border) |
        (candidates[:, _Y] > ny - 1 - border)
    )

    # Size-aware NMS
    picks = size_aware_nms(candidates, beta=cfg.nms_beta, edge_mask=edge_mask)

    # Optional radial refinement
    if cfg.refine and picks.shape[0] > 0:
        from .refine import refine_picks
        picks = refine_picks(
            image, picks, margin=cfg.refine_margin,
            r_min=cfg.dmin / 2.0, r_max=cfg.dmax / 2.0,
        )

        # Post-refine diameter filter: discard picks whose refined diameter
        # falls outside [dmin, dmax].
        diameters = picks[:, _SIGMA] * _SIGMA_TO_DIAM
        keep = (diameters >= cfg.dmin) & (diameters <= cfg.dmax)
        picks = picks[keep]

    # Pixel-statistics filter
    if picks.shape[0] > 0:
        picks = _filter_by_pixel_stats(
            image, picks, max_local_contrast=cfg.max_local_contrast,
        )

    # Anti-cluster filter: reject large picks that contain multiple
    # smaller accepted picks inside them (cluster false positives).
    if picks.shape[0] > 1:
        picks = _filter_cluster_picks(picks, min_interior_picks=2)

    # Post-refinement overlap filter
    if cfg.max_overlap > 0 and picks.shape[0] > 1:
        picks = _filter_overlapping_picks(
            picks, cfg.max_overlap, overlap_mode=cfg.overlap_mode,
        )

    # ── Pass 2: morphological closing + re-detect ──────────────────────
    if cfg.pass2 and picks.shape[0] > 0:
        picks = _run_pass2(image, picks, cfg)

    return picks


def _make_pass2_config(cfg: PickerConfig) -> PickerConfig:
    """Build a PickerConfig for pass-2 detection on the closed image."""
    from .config import _auto_pyramid_levels

    return PickerConfig(
        dmin=cfg.pass2_dmin,
        dmax=cfg.pass2_dmax,
        threshold_percentile=cfg.pass2_threshold_percentile,
        nms_beta=cfg.nms_beta,
        refine=False,         # no refinement for pass-2 — small particles inside
                              # the large faint targets dominate the radial gradient,
                              # causing refinement to shrink diameters drastically;
                              # the DoG scale estimate from the closed image is used
        pixel_size=cfg.pixel_size,
        dog_k=cfg.dog_k,
        dog_min_steps=cfg.dog_min_steps,
        min_score=cfg.min_score,
        edge_fraction=cfg.edge_fraction,
        max_local_contrast=cfg.max_local_contrast,
        max_overlap=0.0,  # overlap filter runs on merged set, not pass-2 alone
        pass2=False,      # no recursive pass-2
    )


def _run_pass2(
    image: np.ndarray,
    picks_pass1: np.ndarray,
    cfg: PickerConfig,
) -> np.ndarray:
    """
    Two-pass detection via morphological closing.

    1. Clean the image: grey closing removes dark features smaller than the SE.
    2. Detect on the cleaned image at large scales (pass-2 config).
    3. Pre-NMS size filter to remove residual small features near the
       closing boundary.
    4. Edge exclusion + NMS.
    5. Merge pass-1 + pass-2 and deduplicate with NMS.

    No pixel-stats or anti-cluster filtering — those filters were designed
    for pass-1 (small dark particles on smooth background) and reject the
    large faint particles pass-2 is meant to find.  Radial refinement runs
    on the closed image to correct the DoG diameter overestimate (the
    closing broadens features, inflating the DoG scale).

    Parameters
    ----------
    image : ndarray, shape (ny, nx), float32
    picks_pass1 : ndarray, shape (N, 5), float32
    cfg : PickerConfig with pass2=True

    Returns
    -------
    merged : ndarray, shape (M, 5), float32
    """
    from .morph import morphological_clean

    # 1. Clean image
    cleaned = morphological_clean(image, se_radius=cfg.closing_radius)

    # 2. Detect on cleaned image
    cfg2 = _make_pass2_config(cfg)
    candidates = _detect_dog(cleaned, cfg2)
    if candidates.shape[0] == 0:
        return picks_pass1

    # 3. Pre-NMS size filter: closing partially attenuates features near the
    #    removal boundary, leaving residual peaks that would suppress the
    #    large targets in NMS.  Remove candidates below 2.5 * closing_radius.
    min_pass2_diam = max(cfg2.dmin, 2.5 * cfg.closing_radius)
    cand_diams = candidates[:, _SIGMA] * _SIGMA_TO_DIAM
    candidates = candidates[cand_diams >= min_pass2_diam]
    if candidates.shape[0] == 0:
        return picks_pass1

    # 4. Edge exclusion + NMS
    ny, nx = cleaned.shape
    border = cfg2.dmax / 2.0 * cfg2.edge_fraction
    edge_mask = (
        (candidates[:, _X] < border) |
        (candidates[:, _X] > nx - 1 - border) |
        (candidates[:, _Y] < border) |
        (candidates[:, _Y] > ny - 1 - border)
    )
    picks2 = size_aware_nms(candidates, beta=cfg2.nms_beta, edge_mask=edge_mask)

    if picks2.shape[0] == 0:
        return picks_pass1

    # 5. Radial refinement on the CLOSED image to correct the DoG diameter
    #    overestimate.  The closed image is smooth (small particles removed),
    #    so refinement finds the true edge of the large particle without
    #    interference from small neighbours.
    if cfg.refine and picks2.shape[0] > 0:
        from .refine import refine_picks
        picks2 = refine_picks(
            cleaned, picks2, margin=0.5,
            r_min=cfg2.dmin / 2.0, r_max=cfg2.dmax / 2.0,
        )
        # Post-refine diameter filter
        diameters = picks2[:, _SIGMA] * _SIGMA_TO_DIAM
        keep = (diameters >= cfg2.dmin) & (diameters <= cfg2.dmax)
        picks2 = picks2[keep]

    # 6. Mode-based local-contrast filter on the CLOSED image.
    #    True large particles are dark islands in a uniformly bright field
    #    (annulus mode = bright), while false positives (inter-particle gaps)
    #    sit in uniformly grey regions (annulus mode ≈ interior mode).
    #    Using the mode (histogram peak) instead of the mean avoids being
    #    pulled by outlier pixels.
    if picks2.shape[0] > 0:
        picks2 = _filter_pass2_by_mode_contrast(cleaned, image, picks2)

    if picks2.shape[0] == 0:
        return picks_pass1

    # 7. Merge pass-1 + pass-2, deduplicate with NMS
    merged = np.vstack([picks_pass1, picks2])
    merged = size_aware_nms(merged, beta=cfg.nms_beta)

    return merged


def _histogram_mode(pixels: np.ndarray, n_bins: int = 64) -> float:
    """Return the mode (most common value) of *pixels* via histogram peak."""
    counts, edges = np.histogram(pixels, bins=n_bins)
    peak_bin = int(np.argmax(counts))
    return float(0.5 * (edges[peak_bin] + edges[peak_bin + 1]))


def _mode_contrast(
    image: np.ndarray,
    cx: float,
    cy: float,
    radius: float,
    r_inner: float = 1.5,
    r_outer: float = 2.5,
) -> float:
    """Compute (outer_ring_mode - interior_mode) / image_std for one pick.

    r_inner and r_outer are fractions of radius defining the reference ring.
    Default (1.5–2.5r) gives a wide annulus.  Use (1.0–1.4r) for the tight
    halo immediately outside the particle edge.
    """
    ny, nx = image.shape
    img_std = float(np.std(image))
    if img_std <= 0:
        return 0.0

    ar = r_outer * radius
    ax0 = max(0, int(cx - ar) - 1)
    ax1 = min(nx, int(cx + ar) + 2)
    ay0 = max(0, int(cy - ar) - 1)
    ay1 = min(ny, int(cy + ar) + 2)

    ayy, axx = np.mgrid[ay0:ay1, ax0:ax1]
    adist_sq = (axx - cx) ** 2 + (ayy - cy) ** 2

    circle_mask = adist_sq <= radius * radius
    if circle_mask.sum() == 0:
        return 0.0
    interior_mode = _histogram_mode(image[ay0:ay1, ax0:ax1][circle_mask])

    ring_mask = (adist_sq > (r_inner * radius) ** 2) & (adist_sq <= ar * ar)
    if ring_mask.sum() == 0:
        return 0.0
    ring_mode = _histogram_mode(image[ay0:ay1, ax0:ax1][ring_mask])

    return (ring_mode - interior_mode) / img_std


def _filter_pass2_by_mode_contrast(
    cleaned: np.ndarray,
    original: np.ndarray,
    picks: np.ndarray,
    min_contrast_closed: float = 3.5,
    min_halo_contrast: float = 1.0,
) -> np.ndarray:
    """Remove pass-2 false positives using mode-based contrast on both images.

    A pick is kept only if it passes BOTH:
      1. Wide-annulus contrast on closed image >= min_contrast_closed
         (large particle = dark island in bright closed-image background)
      2. Tight-halo contrast on original image >= min_halo_contrast
         The halo ring (1.0–1.4r) just outside the particle edge is very
         bright for real large particles (lipid shell + solvent), while
         false positives in inter-particle gaps are surrounded by grey.
    """
    keep = np.ones(len(picks), dtype=bool)

    for i, pick in enumerate(picks):
        cx, cy = float(pick[_X]), float(pick[_Y])
        radius = float(pick[_SIGMA]) * _SIGMA_TO_DIAM / 2.0

        lc_closed = _mode_contrast(cleaned, cx, cy, radius,
                                   r_inner=1.5, r_outer=2.5)
        if lc_closed < min_contrast_closed:
            keep[i] = False
            continue

        halo = _mode_contrast(original, cx, cy, radius,
                              r_inner=1.0, r_outer=1.4)
        if halo < min_halo_contrast:
            keep[i] = False

    return picks[keep]


def _filter_by_pixel_stats(
    image: np.ndarray,
    picks: np.ndarray,
    min_dark_fraction: float = 0.3,
    ambiguous_dark_fraction: float = 0.6,
    min_local_contrast: float = 0.2,
    max_local_contrast: float = 0.0,
) -> np.ndarray:
    """
    Remove false-positive picks using interior darkness and local contrast.

    Three rejection criteria:
      1. dark_fraction < min_dark_fraction → bright interior (inflated spot)
      2. dark_fraction < ambiguous_dark_fraction AND local_contrast < min_local_contrast
         → not clearly dark, and not darker than surroundings
      3. local_contrast > max_local_contrast → isolated contaminant (ice/ethane)
         (only when max_local_contrast > 0)

    dark_fraction: fraction of interior pixels below the global image mean.
    local_contrast: (annulus_mean - interior_mean) / image_std, where the
    annulus covers 1.5r to 2.5r around the pick center.
    """
    ny, nx = image.shape
    img_mean = float(np.mean(image))
    img_std = float(np.std(image))

    keep = np.ones(len(picks), dtype=bool)

    for i, pick in enumerate(picks):
        cx, cy = float(pick[_X]), float(pick[_Y])
        radius = float(pick[_SIGMA]) * _SIGMA_TO_DIAM / 2.0

        # Interior circle
        x0 = max(0, int(cx - radius) - 1)
        x1 = min(nx, int(cx + radius) + 2)
        y0 = max(0, int(cy - radius) - 1)
        y1 = min(ny, int(cy + radius) + 2)

        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
        circle_mask = dist_sq <= radius * radius

        if circle_mask.sum() == 0:
            continue

        pixels = image[y0:y1, x0:x1][circle_mask]
        dark_fraction = float((pixels < img_mean).sum()) / len(pixels)

        # (a) Bright interior → reject
        if dark_fraction < min_dark_fraction:
            keep[i] = False
            continue

        # Compute local contrast (needed for stages b and c)
        interior_mean = float(np.mean(pixels))

        ar = 2.5 * radius
        ax0 = max(0, int(cx - ar) - 1)
        ax1 = min(nx, int(cx + ar) + 2)
        ay0 = max(0, int(cy - ar) - 1)
        ay1 = min(ny, int(cy + ar) + 2)

        ayy, axx = np.mgrid[ay0:ay1, ax0:ax1]
        adist_sq = (axx - cx) ** 2 + (ayy - cy) ** 2
        annulus_mask = (adist_sq > (1.5 * radius) ** 2) & (adist_sq <= ar * ar)

        if annulus_mask.sum() == 0:
            continue

        annulus_mean = float(np.mean(image[ay0:ay1, ax0:ax1][annulus_mask]))
        local_contrast = (annulus_mean - interior_mean) / img_std if img_std > 0 else 0.0

        # (b) Ambiguous darkness + low local contrast → reject
        if dark_fraction < ambiguous_dark_fraction and local_contrast < min_local_contrast:
            keep[i] = False
            continue

        # (c) Isolated contaminant: very high local contrast → reject
        if max_local_contrast > 0 and local_contrast > max_local_contrast:
            keep[i] = False

    return picks[keep]


def _filter_cluster_picks(
    picks: np.ndarray,
    min_interior_picks: int = 2,
) -> np.ndarray:
    """
    Reject large picks that contain smaller accepted picks inside them.

    A real large particle has a smooth interior — it would not contain
    multiple smaller detected particles.  Large picks that do are almost
    certainly cluster false positives (a group of small particles that
    correlated with a large template).

    For each pick, count how many *smaller* picks have their centre inside
    this pick's circle.  If count >= min_interior_picks, reject it.

    Parameters
    ----------
    picks : ndarray, shape (N, 5)
    min_interior_picks : int
        Reject a pick if it contains at least this many smaller picks.

    Returns
    -------
    filtered : ndarray, shape (M, 5)
    """
    if picks.shape[0] <= 1:
        return picks

    radii = picks[:, _SIGMA] * _SIGMA_TO_DIAM / 2.0
    xy = picks[:, :2]  # (N, 2) — x, y columns

    tree = cKDTree(xy)
    keep = np.ones(len(picks), dtype=bool)

    for i in range(len(picks)):
        r_i = radii[i]
        # Find all picks whose centres fall inside this pick's circle
        nearby = tree.query_ball_point(xy[i], r=r_i)
        # Count only picks that are smaller than this one
        n_interior = 0
        for j in nearby:
            if j == i:
                continue
            if radii[j] < r_i:
                n_interior += 1
        if n_interior >= min_interior_picks:
            keep[i] = False

    return picks[keep]


def _circle_overlap_fraction(
    r1: float, r2: float, d: float, mode: str = "smaller",
) -> float:
    """
    Return the overlap area of two circles as a fraction of a reference area.

    Parameters
    ----------
    r1 : float
        Radius of the candidate circle (the one being tested).
    r2 : float
        Radius of the already-accepted circle.
    d : float
        Distance between their centres.
    mode : str
        Which circle's area to use as denominator:
        - ``"smaller"`` — area of the smaller circle (strict, penalises
          large circles that overlap any small one).
        - ``"candidate"`` — area of r1.  A large candidate overlapping
          a few small accepted picks shows low overlap and survives.
        - ``"larger"`` — area of the larger circle (most permissive).

    Returns
    -------
    fraction : float in [0, 1]
    """
    if d <= 0:
        return 1.0
    if d >= r1 + r2:
        return 0.0
    # One circle contained in the other
    if d + min(r1, r2) <= max(r1, r2):
        if mode == "smaller":
            return 1.0
        elif mode == "candidate":
            # Candidate fully inside accepted → fraction = 1.0
            # Accepted fully inside candidate → fraction = r2²/r1²
            if r1 <= r2:
                return 1.0
            return float(np.clip(r2 * r2 / (r1 * r1), 0.0, 1.0))
        else:  # larger
            return float(np.clip(min(r1, r2) ** 2 / max(r1, r2) ** 2, 0.0, 1.0))

    # Standard circle-circle intersection area
    r1_sq, r2_sq, d_sq = r1 * r1, r2 * r2, d * d
    # Clamp for numerical safety
    cos_alpha = np.clip((d_sq + r1_sq - r2_sq) / (2.0 * d * r1), -1.0, 1.0)
    cos_beta = np.clip((d_sq + r2_sq - r1_sq) / (2.0 * d * r2), -1.0, 1.0)
    alpha = float(np.arccos(cos_alpha))
    beta = float(np.arccos(cos_beta))

    area = r1_sq * (alpha - np.sin(2 * alpha) / 2.0) + \
           r2_sq * (beta - np.sin(2 * beta) / 2.0)

    if mode == "smaller":
        denom = np.pi * min(r1, r2) ** 2
    elif mode == "candidate":
        denom = np.pi * r1_sq
    else:  # larger
        denom = np.pi * max(r1, r2) ** 2

    return float(np.clip(area / denom, 0.0, 1.0))


def _filter_overlapping_picks(
    picks: np.ndarray,
    max_overlap: float,
    overlap_mode: str = "candidate",
) -> np.ndarray:
    """
    Greedy post-refinement overlap filter.

    Keeps higher-scoring picks and rejects lower-scoring ones that overlap
    more than *max_overlap* with any already-accepted pick.

    Parameters
    ----------
    picks : ndarray, shape (N, 5)
    max_overlap : float in (0, 1]
    overlap_mode : str
        Passed to ``_circle_overlap_fraction``.  ``"candidate"`` (default)
        divides intersection by the candidate's area, so large picks
        overlapping a few small accepted ones survive.

    Returns
    -------
    filtered : ndarray, shape (M, 5)
    """
    if picks.shape[0] <= 1 or max_overlap <= 0:
        return picks

    # Sort by score descending
    order = np.argsort(picks[:, _SCORE])[::-1]
    sorted_picks = picks[order]

    accepted_xy = []
    accepted_r = []
    accepted_idx = []
    tree = None
    tree_dirty = True

    for i, pick in enumerate(sorted_picks):
        x, y, sigma = float(pick[_X]), float(pick[_Y]), float(pick[_SIGMA])
        r_i = sigma * _SIGMA_TO_DIAM / 2.0

        if len(accepted_xy) == 0:
            accepted_xy.append([x, y])
            accepted_r.append(r_i)
            accepted_idx.append(i)
            tree_dirty = True
            continue

        if tree_dirty:
            tree = cKDTree(np.array(accepted_xy))
            tree_dirty = False

        # Query all accepted picks within r_i + max_accepted_r
        max_r = max(accepted_r)
        query_radius = r_i + max_r
        nearby = tree.query_ball_point([x, y], r=query_radius)

        rejected = False
        for j in nearby:
            dist = float(np.hypot(x - accepted_xy[j][0], y - accepted_xy[j][1]))
            r_j = accepted_r[j]
            frac = _circle_overlap_fraction(r_i, r_j, dist, mode=overlap_mode)
            if frac > max_overlap:
                rejected = True
                break

        if not rejected:
            accepted_xy.append([x, y])
            accepted_r.append(r_i)
            accepted_idx.append(i)
            tree_dirty = True

    if not accepted_idx:
        return np.empty((0, 5), dtype=np.float32)

    return sorted_picks[accepted_idx]


def process_micrograph(
    mic_path: str | Path,
    outdir: str | Path,
    cfg: Optional[PickerConfig] = None,
    verbose: bool = True,
) -> dict:
    """
    Full pipeline for one micrograph: detect → write all outputs.

    Parameters
    ----------
    mic_path : str or Path
    outdir : str or Path
        Directory for outputs (CSV, STAR, JSON, figures).
    cfg : PickerConfig or None (uses defaults)
    verbose : bool

    Returns
    -------
    result : dict
        {"path": str, "n_picks": int, "time_s": float, "picks_csv": str,
         "picks": ndarray shape (N,5)}
    """
    mic_path = Path(mic_path)
    outdir = Path(outdir)
    if cfg is None:
        cfg = PickerConfig()

    t0 = time.perf_counter()

    if verbose:
        print(f"  Reading: {mic_path.name}")
    image = read_micrograph(mic_path)

    # Image stats for QC (computed once while image is in memory)
    p2, p98 = np.percentile(image, (2, 98))
    image_std = float(np.std(image))
    dynamic_range = float(p98 - p2)
    image_shape = image.shape  # (ny, nx)

    picks = pick_micrograph(image, cfg)
    n_picks = picks.shape[0]

    stem = mic_path.stem
    picks_list = _picks_array_to_dicts(picks)

    # Write CSV
    csv_path = None
    if cfg.write_csv:
        csv_path = outdir / f"{stem}_picks.csv"
        write_picks_csv(picks_list, csv_path)

    # Write STAR
    if cfg.write_star and n_picks > 0:
        write_picks_star(picks_list, outdir / f"{stem}_picks.star")

    # Write extraction plan JSON
    plan = None
    if cfg.write_extraction_plan and n_picks > 0:
        diameters = picks[:, _SIGMA] * _SIGMA_TO_DIAM
        plan = make_extraction_plan(diameters, n_bins=cfg.n_size_bins)
        write_extraction_plan(plan, outdir / f"{stem}_extraction_plan.json")

    # Overlay figure
    if cfg.write_overlay:
        plot_picks_overlay(
            image, picks, outdir,
            name=f"{stem}_picks_overlay",
            dmin=cfg.dmin, dmax=cfg.dmax,
            pixel_size=cfg.pixel_size,
            formats=cfg.figure_formats, dpi=cfg.figure_dpi,
        )
        plt_close()

    # Histogram
    if cfg.write_histogram and n_picks > 0:
        plot_size_histogram(
            picks, outdir,
            name=f"{stem}_size_histogram",
            dmin=cfg.dmin, dmax=cfg.dmax,
            pixel_size=cfg.pixel_size,
            extraction_plan=plan,
            formats=cfg.figure_formats, dpi=cfg.figure_dpi,
        )
        plt_close()

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"  {n_picks} picks  ({elapsed:.1f}s)")

    return {
        "path": str(mic_path),
        "n_picks": n_picks,
        "time_s": round(elapsed, 2),
        "picks_csv": str(csv_path) if csv_path else None,
        "picks": picks,
        "image_std": image_std,
        "dynamic_range": dynamic_range,
        "image_shape": image_shape,
    }


def process_batch(
    mic_dir,
    outdir: str | Path,
    cfg: Optional[PickerConfig] = None,
    verbose: bool = True,
    workers: int = 1,
    show_mic: Optional[str] = None,
) -> List[dict]:
    """
    Process all micrographs in a directory (or from a pre-built path list).

    Parameters
    ----------
    mic_dir : str, Path, or list of Path
        Directory containing .mrc / .tif files, or a pre-built list of
        micrograph paths (e.g. gathered from multiple directories).
    outdir : str or Path
        Output directory.
    cfg : PickerConfig or None
    verbose : bool
    workers : int
        Number of parallel workers.  1 = sequential (no multiprocessing
        overhead).  >1 uses ``multiprocessing.Pool``.
    show_mic : str or None
        If set, pin this micrograph (stem name substring) in the summary
        figure as one of the three representative panels.

    Returns
    -------
    results : list of dict
    """
    if isinstance(mic_dir, (list, tuple)):
        mic_paths = sorted(mic_dir)
    else:
        mic_paths = list_micrographs(mic_dir)
    if not mic_paths:
        print(f"No micrographs found in {mic_dir}")
        return []

    if cfg is None:
        cfg = PickerConfig()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"Found {len(mic_paths)} micrograph(s) in {mic_dir}")
        if workers > 1:
            print(f"Using {workers} workers")

    t0_wall = time.perf_counter()

    # ── Per-micrograph processing ────────────────────────────────────────
    if workers > 1:
        args_list = [(str(p), str(outdir), cfg, verbose) for p in mic_paths]
        with multiprocessing.Pool(workers) as pool:
            results = pool.starmap(_process_one, args_list)
    else:
        results = []
        for i, p in enumerate(mic_paths, 1):
            if verbose:
                print(f"[{i}/{len(mic_paths)}] {p.name}")
            result = process_micrograph(p, outdir, cfg=cfg, verbose=verbose)
            results.append(result)

    wall_time = time.perf_counter() - t0_wall
    total_picks = sum(r["n_picks"] for r in results)
    total_cpu = sum(r["time_s"] for r in results)

    if verbose:
        print(f"\nDone. {total_picks} total picks across {len(results)} "
              f"micrographs ({wall_time:.1f}s wall, {total_cpu:.1f}s CPU)")

    # ── QC flagging ─────────────────────────────────────────────────────
    _compute_qc_flags(results)
    n_pass = sum(1 for r in results if r["qc_pass"])
    n_flag = len(results) - n_pass
    if verbose:
        print(f"QC: {n_pass} PASS, {n_flag} FLAGGED")

    # ── Combined CSV ─────────────────────────────────────────────────────
    combined_rows: List[dict] = []
    for r in results:
        stem = Path(r["path"]).stem
        for row in _picks_array_to_dicts(r["picks"]):
            row["micrograph"] = stem
            combined_rows.append(row)
    write_combined_csv(combined_rows, outdir / "all_picks.csv")
    if verbose:
        print(f"Combined CSV: {outdir / 'all_picks.csv'}  ({len(combined_rows)} rows)")

    # ── QC CSV ────────────────────────────────────────────────────────────
    _write_qc_csv(results, outdir / "micrograph_qc.csv")
    if verbose:
        print(f"QC CSV: {outdir / 'micrograph_qc.csv'}")

    # ── Batch summary figure ─────────────────────────────────────────────
    _generate_batch_summary(results, outdir, cfg, wall_time, verbose,
                            show_mic=show_mic)

    return results


# --------------------------------------------------------------------------- #
# Multiprocessing helper (must be module-level for pickling)
# --------------------------------------------------------------------------- #

def _process_one(mic_path: str, outdir: str, cfg: PickerConfig, verbose: bool) -> dict:
    """Thin wrapper around process_micrograph for multiprocessing.Pool.starmap."""
    return process_micrograph(mic_path, outdir, cfg=cfg, verbose=verbose)


# --------------------------------------------------------------------------- #
# QC flagging (MAD-based outlier detection)
# --------------------------------------------------------------------------- #

def _compute_qc_flags(results: List[dict]) -> None:
    """
    Flag bad micrographs using MAD-based outlier detection.

    Adds ``"qc_pass"`` (bool) and ``"qc_flags"`` (list of str) to each result
    dict in-place.

    Metrics:
      - image_std     (high = thick ice)
      - dynamic_range (high = bad ice)
      - pick_density  (low  = empty / bad image)
      - diameter_iqr  (high = heterogeneous sizes)
    """
    for r in results:
        r["qc_pass"] = True
        r["qc_flags"] = []

    if len(results) < 5:
        # Not enough micrographs for meaningful MAD-based flagging
        return

    # Compute per-micrograph metrics
    for r in results:
        ny, nx = r["image_shape"]
        area = float(ny * nx)
        r["_pick_density"] = r["n_picks"] / area * 1e6

        picks = r["picks"]
        if picks.shape[0] > 3:
            diams = picks[:, _SIGMA] * _SIGMA_TO_DIAM
            r["_diameter_iqr"] = float(np.percentile(diams, 75) - np.percentile(diams, 25))
        else:
            r["_diameter_iqr"] = 0.0

    metrics = {
        "image_std":      ("high", 2.5),
        "dynamic_range":  ("high", 2.5),
        "_pick_density":  ("low",  2.5),
        "_diameter_iqr":  ("high", 6.0),
    }

    for metric, (direction, mad_thresh) in metrics.items():
        values = np.array([r[metric] for r in results])
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        if mad < 1e-9:
            continue
        for r in results:
            deviation = (r[metric] - med) / mad
            if direction == "low" and deviation < -mad_thresh:
                r["qc_flags"].append(metric.strip("_"))
                r["qc_pass"] = False
            elif direction == "high" and deviation > mad_thresh:
                r["qc_flags"].append(metric.strip("_"))
                r["qc_pass"] = False


# --------------------------------------------------------------------------- #
# Batch summary helper
# --------------------------------------------------------------------------- #

def _generate_batch_summary(
    results: List[dict],
    outdir: Path,
    cfg: PickerConfig,
    wall_time: float,
    verbose: bool,
    show_mic: Optional[str] = None,
) -> None:
    """Select 3 representative micrographs (QC PASS only) and generate the batch summary figure."""
    # Only pick representatives from QC-passing micrographs with picks
    results_with_picks = [r for r in results
                          if r["n_picks"] > 0 and r.get("qc_pass", True)]
    if not results_with_picks:
        return

    # If show_mic is requested, find it (may be flagged — still show it)
    pinned = None
    if show_mic:
        for r in results:
            if show_mic in Path(r["path"]).stem:
                pinned = r
                break

    # Sort by pick count and select 25th, 50th, 75th percentile micrographs
    sorted_results = sorted(results_with_picks, key=lambda r: r["n_picks"])
    n = len(sorted_results)
    indices = []
    for pct in (0.25, 0.50, 0.75):
        idx = min(int(pct * (n - 1) + 0.5), n - 1)
        if idx not in indices:
            indices.append(idx)

    # Ensure we have up to 3 unique entries (handle n<3)
    while len(indices) < min(3, n):
        for idx in range(n):
            if idx not in indices:
                indices.append(idx)
                break

    labels_map = {0: "Low", 1: "Med", 2: "High"}
    representative_mics = []
    for i, idx in enumerate(indices):
        r = sorted_results[idx]
        label_prefix = labels_map.get(i, "")
        label = f"{label_prefix} ({r['n_picks']} picks)"
        try:
            image = read_micrograph(r["path"])
        except Exception:
            continue
        representative_mics.append((label, image, r["picks"]))

    # Replace the middle panel with the pinned micrograph if requested
    if pinned is not None and len(representative_mics) >= 2:
        try:
            pinned_image = read_micrograph(pinned["path"])
            representative_mics[1] = (
                f"Pinned ({pinned['n_picks']} picks)",
                pinned_image,
                pinned["picks"],
            )
        except Exception:
            pass

    if not representative_mics:
        return

    # Stack all picks into one array
    all_picks_arrays = [r["picks"] for r in results if r["picks"].shape[0] > 0]
    if all_picks_arrays:
        all_picks = np.vstack(all_picks_arrays)
    else:
        all_picks = np.empty((0, 5), dtype=np.float32)

    plot_batch_summary(
        representative_mics,
        all_picks,
        n_micrographs=len(results),
        total_time=wall_time,
        outdir=outdir,
        dmin=cfg.dmin, dmax=cfg.dmax,
        pixel_size=cfg.pixel_size,
        log_scale=cfg.log_scale,
        formats=cfg.figure_formats, dpi=cfg.figure_dpi,
    )
    plt_close()
    if verbose:
        print(f"Summary figure: {outdir / 'batch_summary.png'}")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _picks_array_to_dicts(picks: np.ndarray) -> List[dict]:
    """Convert picks ndarray to list of dicts for CSV/STAR writing."""
    records = []
    for row in picks:
        x, y, sigma, score, ds = row
        diameter = float(sigma) * _SIGMA_TO_DIAM
        records.append({
            "x_px": round(float(x), 2),
            "y_px": round(float(y), 2),
            "diameter_px": round(diameter, 2),
            "score": round(float(score), 6),
            "pyramid_level": int(ds),
            "sigma_px": round(float(sigma), 4),
        })
    return records


def _write_qc_csv(results: List[dict], path: Path) -> None:
    """Write per-micrograph QC metrics to CSV."""
    import csv as csv_mod

    fields = ["micrograph", "n_picks", "image_std", "dynamic_range",
              "pick_density", "diameter_iqr", "qc_pass", "flags"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "micrograph": Path(r["path"]).stem,
                "n_picks": r["n_picks"],
                "image_std": f"{r['image_std']:.6f}",
                "dynamic_range": f"{r['dynamic_range']:.4f}",
                "pick_density": f"{r.get('_pick_density', 0):.2f}",
                "diameter_iqr": f"{r.get('_diameter_iqr', 0):.2f}",
                "qc_pass": r.get("qc_pass", True),
                "flags": "; ".join(r.get("qc_flags", [])),
            })


def plt_close():
    """Close all matplotlib figures to free memory."""
    import matplotlib.pyplot as plt
    plt.close("all")
