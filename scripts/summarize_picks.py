#!/usr/bin/env python3
"""
summarize_picks.py — Regenerate summary outputs from per-micrograph pick CSVs.

Use this to produce all_picks.csv, micrograph_qc.csv, and a summary figure
after a partial or completed MPI run (the MPI runner only writes these at the
very end, so a timeout leaves them missing even though all per-mic CSVs exist).

Works from CSVs alone — no micrograph files needed.  image_std / dynamic_range
QC metrics are skipped (they require pixel data); pick_density and diameter_iqr
QC metrics are computed normally.

Usage (CLI):
    python scripts/summarize_picks.py -o /path/to/outputs/ --pixel-size 3.0341

Usage (Spyder / notebook):
    Edit the CONFIGURATION block below and run directly.
"""
from __future__ import annotations

import argparse
import sys
import math
from pathlib import Path

import numpy as np

# ============================================================
# CONFIGURATION — Edit these for your run (Spyder-friendly)
# ============================================================
OUTDIR       = "/path/to/outputs"   # directory with *_picks.csv files
PIXEL_SIZE   = 3.0341               # Å/px (for nm labels on histogram)
IMAGE_SHAPE  = (1296, 1296)         # (ny, nx) — used for pick_density QC metric
FIGURE_DPI   = 300
FIGURE_FMTS  = ("png", "svg")
# ============================================================

_SIGMA_TO_DIAM = 2.0 * math.sqrt(2.0)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Regenerate summary outputs from partial/complete MPI pick CSVs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--outdir", "-o", type=str, default=None,
                   help="Directory containing *_picks.csv files")
    p.add_argument("--pixel-size", type=float, default=None,
                   help="Pixel size in Å/px")
    p.add_argument("--image-shape", type=int, nargs=2, default=None,
                   metavar=("NY", "NX"),
                   help="Image dimensions in pixels (for pick-density QC)")
    p.add_argument("--dpi", type=int, default=None)
    return p.parse_args(argv)


def _load_per_mic_csvs(outdir: Path):
    """
    Read all *_picks.csv files in outdir (skipping all_picks.csv).

    Returns list of result dicts compatible with _compute_qc_flags.
    """
    csv_files = sorted(outdir.glob("*_picks.csv"))
    # Exclude the combined output if it happens to be present
    csv_files = [f for f in csv_files if f.name != "all_picks.csv"]

    results = []
    for csv_path in csv_files:
        stem = csv_path.stem
        # Strip trailing "_picks" to recover micrograph stem
        if stem.endswith("_picks"):
            mic_stem = stem[: -len("_picks")]
        else:
            mic_stem = stem

        rows = _read_csv(csv_path)
        n = len(rows)

        # Reconstruct picks array: columns [x, y, sigma, score, ds]
        if n > 0:
            picks = np.array(
                [[r["x_px"], r["y_px"], r["sigma_px"], r["score"], r["pyramid_level"]]
                 for r in rows],
                dtype=np.float32,
            )
        else:
            picks = np.empty((0, 5), dtype=np.float32)

        results.append({
            "path": str(outdir / (mic_stem + ".mrc")),  # not used for reading
            "micrograph": mic_stem,
            "n_picks": n,
            "picks": picks,
            "image_std": 0.0,       # stub — will be skipped by MAD check (all equal)
            "dynamic_range": 0.0,   # stub — will be skipped by MAD check (all equal)
            "image_shape": IMAGE_SHAPE,
        })

    return results


def _read_csv(path: Path):
    import csv
    rows = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append({
                "x_px":          float(row["x_px"]),
                "y_px":          float(row["y_px"]),
                "diameter_px":   float(row["diameter_px"]),
                "score":         float(row["score"]),
                "pyramid_level": int(row["pyramid_level"]),
                "sigma_px":      float(row["sigma_px"]),
            })
    return rows


def _make_summary_figure(results, outdir, pixel_size, dpi, fmts):
    """Two-panel figure: diameter histogram + per-micrograph pick count bar chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from lipopick.viz import save_figure

    nm = pixel_size / 10.0

    # Collect diameters
    good = [r for r in results if r["qc_pass"]]
    flagged = [r for r in results if not r["qc_pass"]]

    good_diams_nm    = np.array([r["picks"][:, 2] * _SIGMA_TO_DIAM * nm
                                  for r in good if r["picks"].shape[0] > 0], dtype=object)
    good_diams_nm    = np.concatenate(good_diams_nm) if len(good_diams_nm) else np.array([])
    flagged_diams_nm = np.array([r["picks"][:, 2] * _SIGMA_TO_DIAM * nm
                                  for r in flagged if r["picks"].shape[0] > 0], dtype=object)
    flagged_diams_nm = np.concatenate(flagged_diams_nm) if len(flagged_diams_nm) else np.array([])
    all_diams_nm = (np.concatenate([good_diams_nm, flagged_diams_nm])
                    if len(good_diams_nm) or len(flagged_diams_nm) else np.array([]))

    fig, (ax_hist, ax_bar) = plt.subplots(1, 2, figsize=(14, 5))

    # ── Histogram ──────────────────────────────────────────────────────
    if len(all_diams_nm):
        bins = np.linspace(
            max(0, np.floor(all_diams_nm.min()) - 1),
            np.ceil(all_diams_nm.max()) + 1,
            50,
        )
        if len(good_diams_nm):
            ax_hist.hist(good_diams_nm, bins=bins, color="steelblue",
                         edgecolor="white", linewidth=0.4, label="QC PASS")
        if len(flagged_diams_nm):
            ax_hist.hist(flagged_diams_nm, bins=bins, color="salmon",
                         edgecolor="white", linewidth=0.4, alpha=0.7, label="QC FLAG")
        ref = good_diams_nm if len(good_diams_nm) else all_diams_nm
        med = np.median(ref)
        ax_hist.axvline(med, color="red", linestyle="--", linewidth=1.2,
                        label=f"median = {med:.1f} nm")
        ax_hist.legend(fontsize=8)

    ax_hist.set_xlabel("Diameter (nm)", fontsize=11)
    ax_hist.set_ylabel("Count", fontsize=11)
    n_good_picks  = len(good_diams_nm)
    n_total_picks = len(all_diams_nm)
    n_mics        = len(results)
    n_good_mics   = len(good)
    n_flag_mics   = len(flagged)
    ax_hist.set_title(
        f"Diameter distribution — {n_total_picks:,} picks from {n_mics:,} micrographs\n"
        f"({n_good_picks:,} from {n_good_mics} PASS  |  {n_flag_mics} FLAGGED)",
        fontsize=10,
    )

    # ── Per-micrograph bar chart ───────────────────────────────────────
    counts = [r["n_picks"] for r in results]
    colors = ["steelblue" if r["qc_pass"] else "salmon" for r in results]
    x = np.arange(len(results))
    ax_bar.bar(x, counts, color=colors, alpha=0.85, width=1.0, edgecolor="none")

    # Only label x-axis if few enough micrographs to read
    if len(results) <= 40:
        names = [r["micrograph"][-12:] for r in results]   # last 12 chars
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(names, rotation=60, ha="right", fontsize=5)
    else:
        ax_bar.set_xlabel("Micrograph index (sorted by name)", fontsize=10)
        ax_bar.set_xticks([])

    ax_bar.set_ylabel("Pick count", fontsize=11)
    ax_bar.set_title(
        f"Per-micrograph pick counts  ({n_good_mics} PASS / {n_flag_mics} FLAGGED)\n"
        f"QC metrics: pick_density + diameter_iqr  "
        f"(image_std / dynamic_range skipped — CSV-only mode)",
        fontsize=9,
    )

    from matplotlib.patches import Patch
    ax_bar.legend(
        handles=[Patch(facecolor="steelblue", label="PASS"),
                 Patch(facecolor="salmon",    label="FLAGGED")],
        fontsize=8, loc="upper right",
    )

    fig.suptitle(
        f"lipopick partial-run summary  |  {n_mics:,} micrographs  |  "
        f"{n_total_picks:,} total picks",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, "partial_summary", outdir, formats=fmts, dpi=dpi)
    plt.close(fig)


def main(argv=None):
    args = parse_args(argv)

    # Resolve params (CLI overrides editable constants)
    outdir      = Path(args.outdir)      if args.outdir       else Path(OUTDIR)
    pixel_size  = args.pixel_size        if args.pixel_size   else PIXEL_SIZE
    image_shape = tuple(args.image_shape) if args.image_shape else IMAGE_SHAPE
    dpi         = args.dpi               if args.dpi          else FIGURE_DPI

    # Patch module-level constant so _load_per_mic_csvs uses the right shape
    global IMAGE_SHAPE
    IMAGE_SHAPE = image_shape

    # Make repo importable when run as a script
    ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(ROOT))

    from lipopick.pipeline import _compute_qc_flags, _write_qc_csv, _picks_array_to_dicts
    from lipopick.io import write_combined_csv

    # ── Load ──────────────────────────────────────────────────────────
    results = _load_per_mic_csvs(outdir)
    if not results:
        print(f"ERROR: no *_picks.csv files found in {outdir}", file=sys.stderr)
        sys.exit(1)

    total_picks = sum(r["n_picks"] for r in results)
    print(f"Loaded {len(results):,} micrographs, {total_picks:,} total picks")

    # ── QC ────────────────────────────────────────────────────────────
    _compute_qc_flags(results)
    n_pass = sum(1 for r in results if r["qc_pass"])
    n_flag = len(results) - n_pass
    print(f"QC: {n_pass} PASS, {n_flag} FLAGGED  "
          f"(image_std/dynamic_range skipped — metrics derived from CSVs only)")

    # Diameter stats
    all_sigmas = np.concatenate([r["picks"][:, 2] for r in results
                                  if r["picks"].shape[0] > 0])
    if len(all_sigmas):
        diams = all_sigmas * _SIGMA_TO_DIAM
        nm = pixel_size / 10.0
        print(f"Diameter: {diams.min():.0f}–{diams.max():.0f} px  "
              f"({diams.min()*nm:.1f}–{diams.max()*nm:.1f} nm)  "
              f"median {np.median(diams):.0f} px ({np.median(diams)*nm:.1f} nm)")

    # ── Write all_picks.csv ───────────────────────────────────────────
    combined_rows = []
    for r in results:
        stem = r["micrograph"]
        for row in _picks_array_to_dicts(r["picks"]):
            row["micrograph"] = stem
            combined_rows.append(row)
    all_picks_path = outdir / "all_picks.csv"
    write_combined_csv(combined_rows, all_picks_path)
    print(f"Wrote: {all_picks_path}  ({len(combined_rows):,} rows)")

    # ── Write micrograph_qc.csv ───────────────────────────────────────
    qc_path = outdir / "micrograph_qc.csv"
    _write_qc_csv(results, qc_path)
    print(f"Wrote: {qc_path}")

    # ── Summary figure ────────────────────────────────────────────────
    _make_summary_figure(results, outdir, pixel_size, dpi, FIGURE_FMTS)
    fig_path = outdir / "partial_summary.png"
    print(f"Wrote: {fig_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
