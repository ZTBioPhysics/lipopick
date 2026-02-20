#!/usr/bin/env python3
"""
Aggregate particle picks from all PASS micrographs, bin into size classes,
and export per-bin RELION STAR files with optional coordinate rescaling.

Reads micrograph_qc.csv to identify PASS micrographs, loads their pick CSVs,
runs make_extraction_plan(n_bins=6), plots the histogram, and writes one
.star file per bin under star_bins/.

Usage
-----
    python scripts/size_binning.py                       # uses defaults below
    python scripts/size_binning.py --results-dir /path   # override via CLI

    # Rescale coordinates to unbinned pixel size for extraction:
    python scripts/size_binning.py --target-pixel-size 0.96
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lipopick.extraction import make_extraction_plan, _next_power_of_2
from lipopick.io import rescale_picks, write_picks_star
from lipopick.viz import save_figure

# ── CONFIGURATION (edit for your data) ──────────────────────────────────────
RESULTS_DIR = "outputs/real_data_test"
N_BINS = 6
BIN_MODE = "equal_width"  # "quantile" or "equal_width"
PIXEL_SIZE = 3.0341       # Å/px (picking / denoised micrographs)
TARGET_PIXEL_SIZE = None   # Å/px for extraction; None = same as PIXEL_SIZE
MICROGRAPH_EXT = ".mrc"   # extension appended to micrograph names in STAR
DMIN = 50.0               # px — diameter range for histogram axis
DMAX = 100.0              # px
# ────────────────────────────────────────────────────────────────────────────


def load_pass_picks(results_dir: Path) -> List[dict]:
    """Load full pick dicts from all PASS micrographs.

    Each pick dict gets an added 'micrograph_name' field (the stem from QC).
    """
    qc_path = results_dir / "micrograph_qc.csv"
    if not qc_path.exists():
        raise FileNotFoundError(f"QC file not found: {qc_path}")

    # Identify PASS micrographs
    pass_names = []
    with open(qc_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["qc_pass"].strip() == "True":
                pass_names.append(row["micrograph"].strip())

    print(f"Found {len(pass_names)} PASS micrographs")

    # Collect picks
    all_picks: List[dict] = []
    for name in pass_names:
        picks_path = results_dir / f"{name}_picks.csv"
        if not picks_path.exists():
            print(f"  WARNING: missing {picks_path.name}")
            continue
        with open(picks_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                all_picks.append({
                    "x_px": float(row["x_px"]),
                    "y_px": float(row["y_px"]),
                    "diameter_px": float(row["diameter_px"]),
                    "score": float(row["score"]),
                    "pyramid_level": int(row["pyramid_level"]),
                    "sigma_px": float(row["sigma_px"]),
                    "micrograph_name": name,
                })

    print(f"Loaded {len(all_picks)} total picks from PASS micrographs")
    return all_picks


def assign_picks_to_bins(picks: List[dict], plan: dict) -> dict:
    """Assign each pick to a bin based on diameter_px.

    Returns dict mapping bin_id → list of picks.
    """
    bins_picks = {b["bin_id"]: [] for b in plan["bins"]}
    for p in picks:
        d = p["diameter_px"]
        for b in plan["bins"]:
            if b["d_lo"] <= d < b["d_hi"]:
                bins_picks[b["bin_id"]].append(p)
                break
    return bins_picks


def write_bin_star_files(
    bins_picks: dict,
    plan: dict,
    outdir: Path,
    picking_apix: float,
    target_apix: float,
    micrograph_ext: str,
) -> List[Path]:
    """Write one STAR file per bin, with optional coordinate rescaling.

    Returns list of written file paths.
    """
    px_to_nm = picking_apix / 10.0
    star_dir = outdir / "star_bins"
    star_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for b in plan["bins"]:
        bid = b["bin_id"]
        picks = bins_picks[bid]
        if not picks:
            continue

        # Rescale if target differs from picking frame
        if target_apix != picking_apix:
            picks = rescale_picks(picks, picking_apix, target_apix)

        # Micrograph names with extension
        mic_names = [p["micrograph_name"] + micrograph_ext for p in picks]

        # Filename with nm range
        d_lo_nm = b["d_lo"] * px_to_nm
        d_hi_nm = b["d_hi"] * px_to_nm
        fname = f"bin{bid}_{d_lo_nm:.1f}-{d_hi_nm:.1f}nm.star"
        star_path = star_dir / fname

        write_picks_star(
            picks, star_path,
            pixel_size=target_apix,
            micrograph_names=mic_names,
        )
        written.append(star_path)

    return written


def write_extraction_summary(
    plan: dict,
    outdir: Path,
    picking_apix: float,
    target_apix: float,
) -> Path:
    """Write extraction_summary.json with bin info at both pixel sizes."""
    scale = picking_apix / target_apix
    px_to_nm = picking_apix / 10.0

    summary = {
        "picking_pixel_size_angstrom": picking_apix,
        "target_pixel_size_angstrom": target_apix,
        "scale_factor": round(scale, 4),
        "n_particles": plan["n_particles"],
        "bins": [],
    }

    for b in plan["bins"]:
        d_hi_target = b["d_hi"] * scale
        box_target = _next_power_of_2(math.ceil(1.5 * d_hi_target))

        summary["bins"].append({
            "bin_id": b["bin_id"],
            "d_lo_picking_px": b["d_lo"],
            "d_hi_picking_px": b["d_hi"],
            "d_lo_nm": round(b["d_lo"] * px_to_nm, 1),
            "d_hi_nm": round(b["d_hi"] * px_to_nm, 1),
            "d_lo_target_px": round(b["d_lo"] * scale, 1),
            "d_hi_target_px": round(b["d_hi"] * scale, 1),
            "n_particles": b["n_particles"],
            "box_size_picking_px": b["box_size_px"],
            "box_size_target_px": box_target,
        })

    out_path = outdir / "star_bins" / "extraction_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return out_path


def print_plan(plan: dict, picking_apix: float, target_apix: float) -> None:
    """Pretty-print the extraction plan with both pixel frames."""
    px_to_nm = picking_apix / 10.0
    scale = picking_apix / target_apix
    rescaled = (target_apix != picking_apix)

    print(f"\n{'='*80}")
    print(f"  Extraction Plan — {plan['n_particles']} particles in {len(plan['bins'])} bins")
    if rescaled:
        print(f"  Picking: {picking_apix:.4f} Å/px  →  Target: {target_apix:.4f} Å/px"
              f"  (scale ×{scale:.3f})")
    print(f"{'='*80}")
    print(f"  Diameter range: {plan['diameter_min']:.1f} – {plan['diameter_max']:.1f} px"
          f"  ({plan['diameter_min']*px_to_nm:.1f} – {plan['diameter_max']*px_to_nm:.1f} nm)")
    print(f"  Mean: {plan['diameter_mean']:.1f} px ({plan['diameter_mean']*px_to_nm:.1f} nm)"
          f"   Median: {plan['diameter_median']:.1f} px ({plan['diameter_median']*px_to_nm:.1f} nm)")
    print()

    if rescaled:
        print(f"  {'Bin':>3}  {'d_lo':>7}  {'d_hi':>7}  {'(nm)':>12}"
              f"  {'Count':>6}  {'Box(pick)':>9}  {'Box(tgt)':>9}")
        print(f"  {'---':>3}  {'-'*7:>7}  {'-'*7:>7}  {'-'*12:>12}"
              f"  {'-'*6:>6}  {'-'*9:>9}  {'-'*9:>9}")
        for b in plan["bins"]:
            d_hi_target = b["d_hi"] * scale
            box_target = _next_power_of_2(math.ceil(1.5 * d_hi_target))
            nm_range = f"{b['d_lo']*px_to_nm:.1f}-{b['d_hi']*px_to_nm:.1f}"
            print(f"  {b['bin_id']:>3}  {b['d_lo']:>7.1f}  {b['d_hi']:>7.1f}  "
                  f"{nm_range:>12}  {b['n_particles']:>6}  "
                  f"{b['box_size_px']:>9}  {box_target:>9}")
    else:
        print(f"  {'Bin':>3}  {'d_lo (px)':>10}  {'d_hi (px)':>10}  {'d_lo (nm)':>10}  "
              f"{'d_hi (nm)':>10}  {'Count':>6}  {'Box (px)':>8}")
        print(f"  {'---':>3}  {'-'*10:>10}  {'-'*10:>10}  {'-'*10:>10}  "
              f"{'-'*10:>10}  {'-'*6:>6}  {'-'*8:>8}")
        for b in plan["bins"]:
            print(f"  {b['bin_id']:>3}  {b['d_lo']:>10.1f}  {b['d_hi']:>10.1f}  "
                  f"{b['d_lo']*px_to_nm:>10.1f}  {b['d_hi']*px_to_nm:>10.1f}  "
                  f"{b['n_particles']:>6}  {b['box_size_px']:>8}")
    print(f"{'='*80}\n")


def plot_binned_histogram(
    diameters: np.ndarray,
    plan: dict,
    outdir: Path,
    dmin: float,
    dmax: float,
    pixel_size: float,
) -> Path:
    """Plot diameter histogram with bin boundaries overlaid."""
    px_to_nm = pixel_size / 10.0
    unit = "nm"
    diameters_nm = diameters * px_to_nm
    dmin_nm = dmin * px_to_nm
    dmax_nm = dmax * px_to_nm

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Histogram
    hist_bins = np.linspace(dmin_nm, dmax_nm, 50)
    ax.hist(diameters_nm, bins=hist_bins, color="steelblue", edgecolor="white",
            linewidth=0.5, zorder=2)

    # Bin boundaries
    colors = ["#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#3498db", "#e67e22"]
    for i, b in enumerate(plan["bins"]):
        d_lo_nm = b["d_lo"] * px_to_nm
        d_hi_nm = b["d_hi"] * px_to_nm
        color = colors[i % len(colors)]
        # Shade the bin region
        ax.axvspan(d_lo_nm, d_hi_nm, alpha=0.10, color=color, zorder=1)
        # Vertical line at lower boundary
        ax.axvline(d_lo_nm, color=color, linestyle="--", linewidth=1.2, zorder=3)
        # Label
        ax.text(
            (d_lo_nm + d_hi_nm) / 2, ax.get_ylim()[1] if i == 0 else 0,
            f"Bin {i}\n{b['n_particles']} picks\nbox={b['box_size_px']}px",
            ha="center", va="top", fontsize=7.5, color=color, fontweight="bold",
        )
    # Right edge of last bin
    last_hi = plan["bins"][-1]["d_hi"] * px_to_nm
    ax.axvline(last_hi, color=colors[(len(plan["bins"])-1) % len(colors)],
               linestyle="--", linewidth=1.2, zorder=3)

    ax.set_xlabel(f"Diameter ({unit})", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_xlim(dmin_nm, dmax_nm)
    ax.set_title(
        f"Particle size distribution — {len(diameters)} picks, {len(plan['bins'])} bins",
        fontsize=11,
    )
    fig.tight_layout()

    # Fix text positions after tight_layout sets ylim
    ymax = ax.get_ylim()[1]
    for i, txt in enumerate(ax.texts):
        txt.set_y(ymax * 0.95)

    saved = save_figure(fig, f"size_binning_{len(plan['bins'])}bins", outdir)
    plt.close(fig)
    return saved[0]


def main():
    parser = argparse.ArgumentParser(description="Bin particle picks into size classes")
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help="Directory with pick CSVs and micrograph_qc.csv")
    parser.add_argument("--n-bins", type=int, default=N_BINS)
    parser.add_argument("--bin-mode", default=BIN_MODE,
                        choices=["quantile", "equal_width"],
                        help="Binning strategy: equal_width or quantile")
    parser.add_argument("--pixel-size", type=float, default=PIXEL_SIZE,
                        help="Pixel size in Å/px of picking micrographs")
    parser.add_argument("--target-pixel-size", type=float,
                        default=TARGET_PIXEL_SIZE,
                        help="Pixel size in Å/px for extraction (default: same as --pixel-size)")
    parser.add_argument("--micrograph-ext", default=MICROGRAPH_EXT,
                        help="Extension appended to micrograph names in STAR files")
    parser.add_argument("--dmin", type=float, default=DMIN)
    parser.add_argument("--dmax", type=float, default=DMAX)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    picking_apix = args.pixel_size
    target_apix = args.target_pixel_size if args.target_pixel_size else picking_apix

    # Load full picks (not just diameters)
    picks = load_pass_picks(results_dir)
    if len(picks) == 0:
        print("No picks found — nothing to bin.")
        return

    diameters = np.array([p["diameter_px"] for p in picks], dtype=np.float32)

    # Build extraction plan (in picking pixel frame)
    plan = make_extraction_plan(diameters, n_bins=args.n_bins, bin_mode=args.bin_mode)
    print_plan(plan, picking_apix, target_apix)

    # Plot histogram
    fig_path = plot_binned_histogram(
        diameters, plan, results_dir,
        dmin=args.dmin, dmax=args.dmax, pixel_size=picking_apix,
    )
    print(f"Histogram saved to: {fig_path}")

    # Assign picks to bins and write per-bin STAR files
    bins_picks = assign_picks_to_bins(picks, plan)
    star_paths = write_bin_star_files(
        bins_picks, plan, results_dir,
        picking_apix=picking_apix,
        target_apix=target_apix,
        micrograph_ext=args.micrograph_ext,
    )

    # Write extraction summary JSON
    summary_path = write_extraction_summary(plan, results_dir, picking_apix, target_apix)

    # Print summary
    print(f"\nSTAR files written to: {star_paths[0].parent}/")
    for sp in star_paths:
        n_in_file = len(bins_picks[int(sp.stem.split("_")[0].replace("bin", ""))])
        print(f"  {sp.name:>30s}  ({n_in_file} picks)")
    print(f"\nExtraction summary: {summary_path}")

    if target_apix != picking_apix:
        scale = picking_apix / target_apix
        print(f"\nCoordinates rescaled: {picking_apix:.4f} → {target_apix:.4f} Å/px"
              f" (×{scale:.3f})")


if __name__ == "__main__":
    main()
