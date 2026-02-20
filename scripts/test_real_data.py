#!/usr/bin/env python3
"""
test_real_data.py — Phased evaluation of lipopick on real LDLp micrographs.

Phase 1: Visual triage — sample micrographs, render as contrast-stretched grid
Phase 2: Pilot run — pick particles with tuned parameters
Phase 3: QC scoring — per-micrograph quality metrics, flag bad micrographs
Phase 4: Summary figure — aggregate overlays, histogram, per-micrograph stats

Usage (CLI):
    python scripts/test_real_data.py --mic-dir data/raw/denoised_micrographs/ \\
        --outdir outputs/real_data_test --n-triage 6

Usage (Spyder / notebook):
    Edit the CONFIGURATION block below and run the script directly.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

# ============================================================
# CONFIGURATION — Edit these for your data (Spyder-friendly)
# ============================================================
MIC_DIR = "data/raw/denoised_micrographs"
OUTPUT_DIR = "outputs/real_data_test"
N_TRIAGE = 6          # number of micrographs for visual triage
N_PILOT = 62          # number to process (0 = all available)
SEED = 42

# Picker parameters (tuned for LDLp at 3.0341 Å/px)
DMIN = 50             # min particle diameter (px) — below this is noise, not LP
DMAX = 100            # max particle diameter (px) — post-refine filter caps here
THRESHOLD_PCT = 80.0  # dense particle field needs aggressive threshold
NMS_BETA = 1.3        # >1.0 suppresses satellite/boundary picks around large particles
REFINE = True         # radial edge refinement — corrects DoG diameter underestimation
PIXEL_SIZE_A = 3.0341 # Å/px (from MRC header)

FIGURE_DPI = 300
FIGURE_FORMATS = ("png", "svg")
VERBOSE = True
# ============================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Test lipopick on real LDLp micrographs (phased evaluation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mic-dir", type=str, default=None,
                   help="Directory of denoised micrographs (.mrc)")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory")
    p.add_argument("--n-triage", type=int, default=None,
                   help="Number of micrographs for triage grid")
    p.add_argument("--n-pilot", type=int, default=None,
                   help="Number of micrographs for pilot picking")
    p.add_argument("--seed", type=int, default=None,
                   help="Random seed for reproducibility")
    p.add_argument("--dmin", type=float, default=None,
                   help="Minimum particle diameter (px)")
    p.add_argument("--dmax", type=float, default=None,
                   help="Maximum particle diameter (px)")
    p.add_argument("--threshold-pct", type=float, default=None,
                   help="Percentile for DoG threshold")
    p.add_argument("--nms-beta", type=float, default=None,
                   help="NMS exclusion factor")
    p.add_argument("--pixel-size", type=float, default=None,
                   help="Pixel size in Angstroms")
    p.add_argument("--dpi", type=int, default=None,
                   help="Figure DPI")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress progress output")
    return p.parse_args(argv)


# ------------------------------------------------------------------ #
# Phase 1: Visual triage
# ------------------------------------------------------------------ #

def phase1_triage(mic_paths, n_triage, seed, outdir, pixel_size_a,
                  formats, dpi, verbose=True):
    """Sample micrographs and render a contrast-stretched 2×3 grid."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lipopick.io import read_micrograph
    from lipopick.viz import save_figure

    rng = np.random.default_rng(seed)
    n = min(n_triage, len(mic_paths))
    indices = rng.choice(len(mic_paths), size=n, replace=False)
    indices.sort()
    selected = [mic_paths[i] for i in indices]

    if verbose:
        print(f"\n{'='*60}")
        print(f"PHASE 1: Visual triage — {n} micrographs")
        print(f"{'='*60}")
        for i, p in enumerate(selected):
            print(f"  [{i+1}] {p.name}")

    # Determine grid layout
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]
    axes = axes.flatten()

    for i, mic_path in enumerate(selected):
        if verbose:
            print(f"  Reading {mic_path.name}...")
        image = read_micrograph(mic_path)
        p2, p98 = np.percentile(image, (2, 98))
        axes[i].imshow(image, cmap="gray", vmin=p2, vmax=p98,
                       origin="upper", interpolation="nearest")
        short_name = _short_mic_name(mic_path)
        axes[i].set_title(short_name, fontsize=9)
        axes[i].set_xlabel(f"{image.shape[1]}×{image.shape[0]} px", fontsize=8)
        axes[i].tick_params(labelsize=7)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Phase 1: Visual Triage", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_figure(fig, "triage_grid", outdir, formats=formats, dpi=dpi)
    plt.close(fig)

    if verbose:
        print(f"  Saved: {Path(outdir) / 'triage_grid.png'}")

    return selected


# ------------------------------------------------------------------ #
# Phase 2: Pilot run
# ------------------------------------------------------------------ #

def phase2_pilot(selected_paths, n_pilot, outdir, cfg, pixel_size_a,
                 verbose=True):
    """Run the picker on the pilot subset and collect results."""
    from lipopick.io import read_picks_csv
    from lipopick.pipeline import process_micrograph

    paths_to_run = selected_paths[:n_pilot]
    if verbose:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Pilot run — {len(paths_to_run)} micrographs")
        print(f"  dmin={cfg.dmin}px  dmax={cfg.dmax}px  "
              f"threshold={cfg.threshold_percentile}%  beta={cfg.nms_beta}  "
              f"refine={cfg.refine}")
        print(f"  pyramid_levels={cfg.pyramid_levels}")
        print(f"{'='*60}")

    results = []
    all_picks = []

    for i, mic_path in enumerate(paths_to_run, 1):
        if verbose:
            print(f"\n[{i}/{len(paths_to_run)}] {mic_path.name}")

        result = process_micrograph(mic_path, outdir, cfg=cfg, verbose=verbose)
        results.append(result)

        # Load picks back for aggregate analysis
        picks = []
        if result["picks_csv"] and Path(result["picks_csv"]).exists():
            picks = read_picks_csv(result["picks_csv"])

        all_picks.append({
            "name": mic_path.stem,
            "short_name": _short_mic_name(mic_path),
            "path": mic_path,
            "picks": picks,
            "n_picks": result["n_picks"],
            "image_std": result["image_std"],
        })

    # Print summary
    if verbose:
        times = [r["time_s"] for r in results]
        total_picks = sum(r["n_picks"] for r in results)
        total_time = sum(times)
        print(f"\n{'─'*60}")
        print(f"Pilot Summary: {len(results)} micrographs, "
              f"{total_picks} total picks, {total_time:.1f}s total")
        print(f"  Time per image: {np.mean(times):.2f}s mean, "
              f"{np.min(times):.2f}s min, {np.max(times):.2f}s max")
        print(f"{'─'*60}")
        nm_per_px = pixel_size_a / 10.0
        for entry, result in zip(all_picks, results):
            diams = [p["diameter_px"] for p in entry["picks"]]
            if diams:
                d_arr = np.array(diams)
                print(f"  {entry['short_name']:>30s}: "
                      f"{entry['n_picks']:4d} picks, "
                      f"diam {d_arr.min():.0f}–{d_arr.max():.0f} px "
                      f"({d_arr.min()*nm_per_px:.1f}–{d_arr.max()*nm_per_px:.1f} nm), "
                      f"median {np.median(d_arr):.0f} px "
                      f"({np.median(d_arr)*nm_per_px:.1f} nm), "
                      f"{result['time_s']:.1f}s")
            else:
                print(f"  {entry['short_name']:>30s}: 0 picks, {result['time_s']:.1f}s")

    return results, all_picks


# ------------------------------------------------------------------ #
# Phase 3: QC scoring and flagging
# ------------------------------------------------------------------ #

def phase3_qc(all_picks, outdir, dmin, dmax, pixel_size_a, verbose=True):
    """
    Compute per-micrograph quality metrics and flag bad micrographs.

    Metrics used for flagging:
    - image_std:      pixel intensity std (high = thick ice / overlapping particles)
    - dynamic_range:  p98 - p2 of pixel values (high = extreme contrast = bad ice)
    - pick_density:   n_picks / image_area (low = bad ice, empty, or overlapping)
    - diameter_iqr:   IQR of pick diameters (high = heterogeneous = bad)

    A micrograph is flagged if it's an outlier (>2 MAD) on any metric.
    """
    if verbose:
        print(f"\n{'='*60}")
        print("PHASE 3: Micrograph QC scoring")
        print(f"{'='*60}")

    if not all_picks:
        print("  No picks to evaluate.")
        return all_picks

    # Collect all diameters across the dataset for reference
    all_diams = []
    for entry in all_picks:
        for p in entry["picks"]:
            all_diams.append(p["diameter_px"])
    all_diams = np.array(all_diams)

    if len(all_diams) == 0:
        return all_picks

    dataset_median_d = float(np.median(all_diams))

    # Assume square images (1296×1296 for this dataset)
    image_area = 1296.0 * 1296.0

    # Compute per-micrograph metrics
    from lipopick.io import read_micrograph

    for entry in all_picks:
        diams = np.array([p["diameter_px"] for p in entry["picks"]])
        n = len(diams)

        # Image intensity std — high = thick ice, overlapping particles
        entry["image_std_val"] = entry["image_std"]

        # Dynamic range (p98 - p2) — need to read image for percentiles
        image = read_micrograph(entry["path"])
        p2, p98 = np.percentile(image, (2, 98))
        entry["dynamic_range"] = float(p98 - p2)
        del image

        # Pick density (picks per Mpx)
        entry["pick_density"] = n / image_area * 1e6

        # Diameter IQR — high = heterogeneous particle sizes = overlapping / bad
        if n > 3:
            entry["diameter_iqr"] = float(np.percentile(diams, 75) - np.percentile(diams, 25))
        else:
            entry["diameter_iqr"] = 0.0

    # Flag outliers using MAD (median absolute deviation)
    # Per-metric MAD thresholds.  diameter_iqr uses a higher threshold (6.0)
    # because mild IQR elevation is normal biological variation, not bad ice.
    metrics_to_flag = {
        "image_std_val":  ("high", 2.5),
        "dynamic_range":  ("high", 2.5),
        "pick_density":   ("low",  2.5),
        "diameter_iqr":   ("high", 6.0),
    }

    for entry in all_picks:
        entry["flags"] = []
        entry["qc_pass"] = True

    for metric, (direction, mad_thresh) in metrics_to_flag.items():
        values = np.array([e[metric] for e in all_picks])
        med = np.median(values)
        mad = np.median(np.abs(values - med))
        if mad < 1e-9:
            continue

        for entry in all_picks:
            v = entry[metric]
            deviation = (v - med) / mad
            if direction == "low" and deviation < -mad_thresh:
                entry["flags"].append(f"{metric}={v:.3f} (low, {deviation:.1f} MAD)")
                entry["qc_pass"] = False
            elif direction == "high" and deviation > mad_thresh:
                entry["flags"].append(f"{metric}={v:.3f} (high, +{deviation:.1f} MAD)")
                entry["qc_pass"] = False

    # Print QC results
    if verbose:
        nm_per_px = pixel_size_a / 10.0
        print(f"  Dataset median diameter: {dataset_median_d:.0f} px "
              f"({dataset_median_d*nm_per_px:.1f} nm)")
        print()
        print(f"  {'Micrograph':>20s}  {'img_std':>8s}  {'dyn_rng':>8s}  "
              f"{'density':>8s}  {'d_IQR':>8s}  {'QC':>6s}")
        print(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*6}")
        for entry in all_picks:
            status = "PASS" if entry["qc_pass"] else "FLAG"
            print(f"  {entry['short_name']:>20s}  "
                  f"{entry['image_std_val']:8.3f}  "
                  f"{entry['dynamic_range']:8.2f}  "
                  f"{entry['pick_density']:8.1f}  "
                  f"{entry['diameter_iqr']:8.1f}  "
                  f"{status:>6s}")
            if entry["flags"]:
                for flag in entry["flags"]:
                    print(f"  {'':>20s}    → {flag}")

        n_good = sum(1 for e in all_picks if e["qc_pass"])
        n_flag = sum(1 for e in all_picks if not e["qc_pass"])
        print(f"\n  Result: {n_good} PASS, {n_flag} FLAGGED")

    # Write QC CSV
    qc_csv_path = Path(outdir) / "micrograph_qc.csv"
    _write_qc_csv(all_picks, qc_csv_path)
    if verbose:
        print(f"  Saved: {qc_csv_path}")

    return all_picks


def _write_qc_csv(all_picks, path):
    """Write per-micrograph QC metrics to CSV."""
    fields = ["micrograph", "short_name", "n_picks", "image_std", "dynamic_range",
              "pick_density", "diameter_iqr", "qc_pass", "flags"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for entry in all_picks:
            writer.writerow({
                "micrograph": entry["name"],
                "short_name": entry["short_name"],
                "n_picks": entry["n_picks"],
                "image_std": f"{entry['image_std_val']:.6f}",
                "dynamic_range": f"{entry['dynamic_range']:.4f}",
                "pick_density": f"{entry['pick_density']:.2f}",
                "diameter_iqr": f"{entry['diameter_iqr']:.2f}",
                "qc_pass": entry["qc_pass"],
                "flags": "; ".join(entry["flags"]) if entry["flags"] else "",
            })


# ------------------------------------------------------------------ #
# Phase 4: Summary figure
# ------------------------------------------------------------------ #

def phase4_summary(all_picks, outdir, dmin, dmax, pixel_size_a,
                   formats, dpi, verbose=True):
    """Create aggregate summary figure with overlays, histogram, and stats."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lipopick.io import read_micrograph
    from lipopick.viz import save_figure
    from matplotlib.colors import Normalize
    import matplotlib.patches as mpatches

    if verbose:
        print(f"\n{'='*60}")
        print("PHASE 4: Summary figure")
        print(f"{'='*60}")

    if not all_picks:
        print("  No picks to summarise.")
        return

    nm_per_px = pixel_size_a / 10.0

    # Separate good vs flagged
    good = [e for e in all_picks if e["qc_pass"]]
    flagged = [e for e in all_picks if not e["qc_pass"]]

    # Sort good by pick count for representative selection
    sorted_good = sorted(good, key=lambda e: e["n_picks"])
    n = len(sorted_good)

    # Select representatives from GOOD micrographs
    if n >= 3:
        representatives = [
            sorted_good[0],
            sorted_good[n // 2],
            sorted_good[-1],
        ]
    elif n >= 1:
        representatives = sorted_good[:min(n, 3)]
    else:
        # All flagged — show flagged ones instead
        representatives = sorted(flagged, key=lambda e: e["n_picks"])[:3]

    n_rep = max(len(representatives), 1)
    n_cols = max(n_rep, 3)  # at least 3 columns for bottom row

    fig = plt.figure(figsize=(6 * n_cols, 12))
    gs = fig.add_gridspec(2, n_cols, hspace=0.3, wspace=0.25)

    # Top row: overlay panels
    factor = 2.0 * 2.0 ** 0.5
    norm = Normalize(vmin=dmin * nm_per_px, vmax=dmax * nm_per_px)
    cm = plt.get_cmap("plasma")

    for i, entry in enumerate(representatives):
        ax = fig.add_subplot(gs[0, i])
        image = read_micrograph(entry["path"])
        p2, p98 = np.percentile(image, (2, 98))
        ax.imshow(image, cmap="gray", vmin=p2, vmax=p98,
                  origin="upper", interpolation="nearest")

        for pick in entry["picks"]:
            diam = pick["diameter_px"]
            radius = diam / 2.0
            color = cm(norm(diam * nm_per_px))
            circle = mpatches.Circle(
                (pick["x_px"], pick["y_px"]), radius=radius,
                fill=False, edgecolor=color, linewidth=0.8, alpha=0.85,
            )
            ax.add_patch(circle)

        # Label with QC status
        if i == 0:
            label = "few"
        elif i == len(representatives) - 1:
            label = "many"
        else:
            label = "typical"
        qc_tag = "PASS" if entry["qc_pass"] else "FLAGGED"
        ax.set_title(f"{entry['short_name']}\n{entry['n_picks']} picks ({label}) — {qc_tag}",
                     fontsize=9,
                     color="green" if entry["qc_pass"] else "red")
        ax.set_xlim(0, image.shape[1])
        ax.set_ylim(image.shape[0], 0)
        ax.tick_params(labelsize=7)

    # Hide unused top-row axes
    for j in range(len(representatives), n_cols):
        fig.add_subplot(gs[0, j]).set_visible(False)

    # Collect diameters from GOOD micrographs only (convert to nm)
    good_diams_nm = []
    for entry in good:
        for pick in entry["picks"]:
            good_diams_nm.append(pick["diameter_px"] * nm_per_px)
    good_diams_nm = np.array(good_diams_nm) if good_diams_nm else np.array([])

    # Also collect ALL diameters for comparison
    all_diams_nm = []
    for entry in all_picks:
        for pick in entry["picks"]:
            all_diams_nm.append(pick["diameter_px"] * nm_per_px)
    all_diams_nm = np.array(all_diams_nm)

    # Auto-scale histogram range (in nm)
    dmin_nm = dmin * nm_per_px
    dmax_nm = dmax * nm_per_px
    if len(all_diams_nm) > 0:
        hist_lo = max(0, np.floor(all_diams_nm.min()) - 1)
        hist_hi = np.ceil(all_diams_nm.max()) + 1
    else:
        hist_lo, hist_hi = dmin_nm, dmax_nm

    # Bottom-left: diameter histogram (good only, with all overlaid)
    ax_hist = fig.add_subplot(gs[1, :2])

    if len(all_diams_nm) > 0:
        bins = np.linspace(hist_lo, hist_hi, 35)
        if len(good_diams_nm) > 0:
            ax_hist.hist(good_diams_nm, bins=bins, color="steelblue",
                         edgecolor="white", linewidth=0.5, label="Good micrographs")
        if len(flagged) > 0:
            flagged_diams = []
            for entry in flagged:
                for pick in entry["picks"]:
                    flagged_diams.append(pick["diameter_px"] * nm_per_px)
            if flagged_diams:
                ax_hist.hist(flagged_diams, bins=bins, color="salmon",
                             edgecolor="white", linewidth=0.5, alpha=0.6,
                             label="Flagged micrographs")
        ax_hist.set_xlabel("Diameter (nm)", fontsize=11)
        ax_hist.set_ylabel("Count", fontsize=11)
        ax_hist.set_xlim(hist_lo, hist_hi)

        ref_diams = good_diams_nm if len(good_diams_nm) > 0 else all_diams_nm
        median_d = np.median(ref_diams)
        ax_hist.axvline(median_d, color="red", linestyle="--", linewidth=1.2,
                        label=f"median = {median_d:.1f} nm")
        ax_hist.legend(fontsize=8)

    n_good_picks = len(good_diams_nm)
    n_total_picks = len(all_diams_nm)
    ax_hist.set_title(f"Diameter distribution — {n_good_picks} good picks "
                      f"({n_total_picks} total) from {len(all_picks)} micrographs",
                      fontsize=10)

    # Bottom-right: bar chart with QC status colour-coding
    ax_bar = fig.add_subplot(gs[1, 2])

    names = [e["short_name"] for e in all_picks]
    counts = [e["n_picks"] for e in all_picks]
    colors_bar = ["steelblue" if e["qc_pass"] else "salmon" for e in all_picks]
    medians_nm = []
    for entry in all_picks:
        diams = [p["diameter_px"] * nm_per_px for p in entry["picks"]]
        medians_nm.append(np.median(diams) if diams else np.nan)

    x_pos = np.arange(len(names))
    ax_bar.bar(x_pos, counts, color=colors_bar, alpha=0.8)
    ax_bar.set_ylabel("Pick count", fontsize=10, color="steelblue")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax_bar.tick_params(axis="y", labelcolor="steelblue")

    ax_scat = ax_bar.twinx()
    valid_medians = [m for m in medians_nm if not np.isnan(m)]
    marker_colors = ["green" if e["qc_pass"] else "red" for e in all_picks]
    ax_scat.scatter(x_pos, medians_nm, c=marker_colors, marker="D", s=40,
                    zorder=5, edgecolors="black", linewidths=0.5)
    ax_scat.set_ylabel("Median diameter (nm)", fontsize=10, color="red")
    ax_scat.tick_params(axis="y", labelcolor="red")

    if valid_medians:
        pad = max((max(valid_medians) - min(valid_medians)) * 0.3, 1.5)
        ax_scat.set_ylim(min(valid_medians) - pad, max(valid_medians) + pad)

    # Legend for QC status
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue", alpha=0.8, label="PASS"),
        Patch(facecolor="salmon", alpha=0.8, label="FLAGGED"),
    ]
    ax_bar.legend(handles=legend_elements, fontsize=8, loc="upper left")
    ax_bar.set_title("Per-micrograph picks & QC", fontsize=10)

    # Handle extra columns if n_cols > 3
    for j in range(3, n_cols):
        fig.add_subplot(gs[1, j]).set_visible(False)

    n_good_mics = len(good)
    n_flag_mics = len(flagged)
    fig.suptitle(f"lipopick — Real Data Summary (LDLp)  |  "
                 f"{n_good_mics} PASS, {n_flag_mics} FLAGGED",
                 fontsize=14, fontweight="bold")
    save_figure(fig, "real_data_summary", outdir, formats=formats, dpi=dpi)
    plt.close(fig)

    summary_path = Path(outdir) / "real_data_summary.png"
    if verbose:
        print(f"  Saved: {summary_path}")

    return summary_path


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _short_mic_name(mic_path):
    """Extract a short label from the long micrograph filename."""
    name = mic_path.stem
    parts = name.split("_")
    for i, part in enumerate(parts):
        if part == "FoilHole" and i + 1 < len(parts):
            return f"FoilHole_{parts[i+1]}"
    return name[:20]


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main(argv=None):
    args = parse_args(argv)

    # Resolve parameters: CLI overrides editable constants
    mic_dir = Path(args.mic_dir) if args.mic_dir else Path(MIC_DIR)
    outdir = Path(args.outdir) if args.outdir else Path(OUTPUT_DIR)
    n_triage = args.n_triage if args.n_triage is not None else N_TRIAGE
    n_pilot = args.n_pilot if args.n_pilot is not None else N_PILOT
    seed = args.seed if args.seed is not None else SEED
    dmin = args.dmin if args.dmin is not None else DMIN
    dmax = args.dmax if args.dmax is not None else DMAX
    threshold_pct = args.threshold_pct if args.threshold_pct is not None else THRESHOLD_PCT
    nms_beta = args.nms_beta if args.nms_beta is not None else NMS_BETA
    pixel_size_a = args.pixel_size if args.pixel_size is not None else PIXEL_SIZE_A

    dpi = args.dpi if args.dpi is not None else FIGURE_DPI
    verbose = not args.quiet
    formats = tuple(FIGURE_FORMATS)

    # Lazy import
    try:
        from lipopick import PickerConfig
        from lipopick.io import list_micrographs
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from lipopick import PickerConfig
        from lipopick.io import list_micrographs

    # Discover micrographs
    mic_paths = list_micrographs(mic_dir)
    if not mic_paths:
        print(f"ERROR: No micrographs found in {mic_dir}", file=sys.stderr)
        sys.exit(1)

    if verbose:
        print(f"lipopick real-data test")
        print(f"  Micrographs: {len(mic_paths)} in {mic_dir}")
        print(f"  Output:      {outdir}")
        print(f"  Pixel size:  {pixel_size_a} Å/px")
        nm_per_px = pixel_size_a / 10.0
        print(f"  Diameter range: {dmin}–{dmax} px "
              f"({dmin*nm_per_px:.1f}–{dmax*nm_per_px:.1f} nm)")

    outdir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Triage ──
    t0 = time.perf_counter()
    selected = phase1_triage(
        mic_paths, n_triage, seed, outdir, pixel_size_a,
        formats=formats, dpi=dpi, verbose=verbose,
    )

    # ── Phase 2: Pilot ──
    # Disable per-micrograph figures when running many images (summary still generated)
    many = (n_pilot > 10)
    cfg = PickerConfig(
        dmin=dmin,
        dmax=dmax,
        pyramid_levels=((1, float(dmin), float(dmax)),),
        threshold_percentile=threshold_pct,
        nms_beta=nms_beta,
        refine=REFINE,
        pixel_size=pixel_size_a,
        write_csv=True,
        write_star=False,
        write_overlay=not many,
        write_histogram=not many,
        write_extraction_plan=True,
        figure_dpi=dpi,
        figure_formats=formats,
    )

    # Use all micrographs if n_pilot exceeds triage selection
    pilot_paths = mic_paths if n_pilot > len(selected) else selected
    results, all_picks = phase2_pilot(
        pilot_paths, n_pilot, outdir, cfg, pixel_size_a, verbose=verbose,
    )

    # ── Phase 3: QC scoring ──
    all_picks = phase3_qc(
        all_picks, outdir, dmin, dmax, pixel_size_a, verbose=verbose,
    )

    # ── Phase 4: Summary ──
    summary_path = phase4_summary(
        all_picks, outdir, dmin, dmax, pixel_size_a,
        formats=formats, dpi=dpi, verbose=verbose,
    )

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"\n{'='*60}")
        print(f"Done in {elapsed:.1f}s")
        print(f"Outputs in: {outdir}")
        print(f"{'='*60}")

    # Open summary figure on macOS
    if summary_path and summary_path.exists():
        import subprocess
        subprocess.run(["open", str(summary_path)], check=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
