"""
Orchestrator: ties all stages together into pick_micrograph() and process_batch().

Stages per micrograph (pick_micrograph):
  1. Build pyramid levels (downsample with anti-aliasing)
  2. DoG scale-space + local maxima per level
  3. Merge candidates across levels → full-res coordinates
  4. Edge exclusion (reject picks within dmax/2 of border)
  5. Size-aware NMS
  6. Optional radial refinement
  7. Post-refine diameter filter (clip to [dmin, dmax])
  8. Pixel-statistics filter (bright interior / ambiguous / contaminant)
  9. Post-refinement overlap filter (greedy, score-ordered)
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
    all_candidates: List[np.ndarray] = []

    for ds, d_lo, d_hi in cfg.pyramid_levels:
        # 1. Build pyramid level
        level_img = build_pyramid_level(image, ds)

        # 2. Compute DoG sigma range for this level
        sigma_min, sigma_max, n_steps = sigma_range_for_level(
            d_lo, d_hi, ds, cfg.dog_k, cfg.dog_min_steps
        )

        # 3. Compute DoG stack and find local maxima
        dog_stack, sigmas = compute_dog_stack(level_img, sigma_min, n_steps, k=cfg.dog_k)
        # Exclude border pixels (in level-coords) from threshold computation.
        # Border = min(sigma_max * 2, level_image // 4) so we always keep ≥50% interior.
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

        # Free DoG stack memory before next level
        del dog_stack

    # 4. Merge all candidates
    if not all_candidates or all(c.shape[0] == 0 for c in all_candidates):
        return np.empty((0, 5), dtype=np.float32)

    candidates = np.vstack([c for c in all_candidates if c.shape[0] > 0])

    # 5. Edge exclusion: reject within dmax/2 of image border
    border = cfg.dmax / 2.0 * cfg.edge_fraction
    edge_mask = (
        (candidates[:, _X] < border) |
        (candidates[:, _X] > nx - 1 - border) |
        (candidates[:, _Y] < border) |
        (candidates[:, _Y] > ny - 1 - border)
    )

    # 6. Size-aware NMS
    picks = size_aware_nms(candidates, beta=cfg.nms_beta, edge_mask=edge_mask)

    # 7. Optional radial refinement
    if cfg.refine and picks.shape[0] > 0:
        from .refine import refine_picks
        picks = refine_picks(
            image, picks, margin=cfg.refine_margin,
            r_min=cfg.dmin / 2.0, r_max=cfg.dmax / 2.0,
        )

        # 8. Post-refine diameter filter: discard picks whose refined diameter
        # falls outside [dmin, dmax].  Refinement can push diameters beyond the
        # search range when it locks onto a cluster edge or ice boundary.
        diameters = picks[:, _SIGMA] * _SIGMA_TO_DIAM
        keep = (diameters >= cfg.dmin) & (diameters <= cfg.dmax)
        picks = picks[keep]

    # 9. Pixel-statistics filter: remove artifacts based on interior darkness
    # and local contrast.  Three rejection criteria:
    #   (a) dark_fraction < 0.3 → bright interior (inflated small spot)
    #   (b) dark_fraction < 0.6 AND local_contrast < 0.2 → ambiguous darkness
    #       with no contrast against local background
    #   (c) local_contrast > max_local_contrast → isolated contaminant
    if picks.shape[0] > 0:
        picks = _filter_by_pixel_stats(
            image, picks, max_local_contrast=cfg.max_local_contrast,
        )

    # 10. Post-refinement overlap filter: NMS runs before refinement, but
    # refinement can inflate diameters up to 2×, creating new overlaps.
    if cfg.max_overlap > 0 and picks.shape[0] > 1:
        picks = _filter_overlapping_picks(picks, cfg.max_overlap)

    return picks


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


def _circle_overlap_fraction(r1: float, r2: float, d: float) -> float:
    """
    Return the overlap area of two circles divided by the area of the smaller.

    Parameters
    ----------
    r1, r2 : float
        Radii of the two circles.
    d : float
        Distance between their centres.

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
        return 1.0

    # Standard circle-circle intersection area
    r1_sq, r2_sq, d_sq = r1 * r1, r2 * r2, d * d
    # Clamp for numerical safety
    cos_alpha = np.clip((d_sq + r1_sq - r2_sq) / (2.0 * d * r1), -1.0, 1.0)
    cos_beta = np.clip((d_sq + r2_sq - r1_sq) / (2.0 * d * r2), -1.0, 1.0)
    alpha = float(np.arccos(cos_alpha))
    beta = float(np.arccos(cos_beta))

    area = r1_sq * (alpha - np.sin(2 * alpha) / 2.0) + \
           r2_sq * (beta - np.sin(2 * beta) / 2.0)

    smaller_area = np.pi * min(r1, r2) ** 2
    return float(np.clip(area / smaller_area, 0.0, 1.0))


def _filter_overlapping_picks(
    picks: np.ndarray,
    max_overlap: float,
) -> np.ndarray:
    """
    Greedy post-refinement overlap filter.

    Keeps higher-scoring picks and rejects lower-scoring ones that overlap
    more than *max_overlap* (as a fraction of the smaller circle) with any
    already-accepted pick.

    Parameters
    ----------
    picks : ndarray, shape (N, 5)
    max_overlap : float in (0, 1]

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
            frac = _circle_overlap_fraction(r_i, r_j, dist)
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
