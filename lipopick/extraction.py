"""
Extraction plan generator.

Given a set of particle diameter estimates, suggests:
  - N size bins (equal-count quantiles or equal-width)
  - Recommended box size for each bin (next power of 2 above 1.5 × d_max_in_bin)
  - Pick counts per bin

Output: dict suitable for JSON serialisation.
"""
from __future__ import annotations

import math
from typing import List

import numpy as np


def make_extraction_plan(
    diameters: np.ndarray,
    n_bins: int = 3,
    box_padding: float = 1.5,
    bin_mode: str = "quantile",
    min_bin_count: int = 0,
) -> dict:
    """
    Build an extraction plan from particle diameter estimates.

    Parameters
    ----------
    diameters : ndarray, shape (N,), float
        Per-particle diameter estimates (pixels).
    n_bins : int
        Desired number of extraction size classes.
    box_padding : float
        Box size = next_power_of_2(box_padding * bin_dmax).
    bin_mode : {"quantile", "equal_width"}
        "quantile" — equal-count bins (default).
        "equal_width" — evenly spaced bins from dmin to dmax.
    min_bin_count : int
        If > 0, merge tail bins (from the right) until the last bin has at
        least this many particles.  Bins are merged pairwise from the right
        until the condition is met or only one bin remains.

    Returns
    -------
    plan : dict
        {
          "n_particles": int,
          "diameter_min": float,
          "diameter_max": float,
          "diameter_mean": float,
          "diameter_median": float,
          "bins": [
            {
              "bin_id": 0,
              "d_lo": float,
              "d_hi": float,
              "d_center": float,
              "n_particles": int,
              "box_size_px": int,   # next power of 2 ≥ padding * d_hi
            },
            ...
          ]
        }
    """
    diameters = np.asarray(diameters, dtype=np.float32)
    diameters = diameters[np.isfinite(diameters) & (diameters > 0)]

    if diameters.size == 0:
        return {"n_particles": 0, "bins": []}

    # Bin edges
    if bin_mode == "equal_width":
        edges = np.linspace(diameters.min(), diameters.max() * 1.001, n_bins + 1)
    else:
        edges = np.percentile(diameters, np.linspace(0, 100, n_bins + 1))
        edges[0] = diameters.min()
        edges[-1] = diameters.max() * 1.001   # ensure last particle included

    bins = []
    for b in range(n_bins):
        d_lo = float(edges[b])
        d_hi = float(edges[b + 1])
        mask = (diameters >= d_lo) & (diameters < d_hi)
        n_in_bin = int(mask.sum())
        d_center = float(np.median(diameters[mask])) if n_in_bin > 0 else (d_lo + d_hi) / 2

        box_raw = box_padding * d_hi
        box_size = _next_power_of_2(int(math.ceil(box_raw)))

        bins.append({
            "bin_id": b,
            "d_lo": round(d_lo, 1),
            "d_hi": round(d_hi, 1),
            "d_center": round(d_center, 1),
            "n_particles": n_in_bin,
            "box_size_px": box_size,
        })

    # Merge tail bins until the last bin meets min_bin_count
    if min_bin_count > 0:
        while len(bins) > 1 and bins[-1]["n_particles"] < min_bin_count:
            last = bins.pop()
            prev = bins[-1]
            merged_d_hi  = last["d_hi"]
            merged_n     = prev["n_particles"] + last["n_particles"]
            merged_mask  = (diameters >= prev["d_lo"]) & (diameters < merged_d_hi)
            merged_center = (float(np.median(diameters[merged_mask]))
                             if merged_n > 0 else (prev["d_lo"] + merged_d_hi) / 2)
            merged_box   = _next_power_of_2(int(math.ceil(box_padding * merged_d_hi)))
            bins[-1] = {
                "bin_id":      prev["bin_id"],
                "d_lo":        prev["d_lo"],
                "d_hi":        round(merged_d_hi, 1),
                "d_center":    round(merged_center, 1),
                "n_particles": merged_n,
                "box_size_px": merged_box,
            }

    plan = {
        "n_particles": int(diameters.size),
        "diameter_min": float(round(diameters.min(), 1)),
        "diameter_max": float(round(diameters.max(), 1)),
        "diameter_mean": float(round(diameters.mean(), 1)),
        "diameter_median": float(round(np.median(diameters), 1)),
        "bins": bins,
    }
    return plan


def _next_power_of_2(n: int) -> int:
    """Return the smallest power of 2 that is >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()
