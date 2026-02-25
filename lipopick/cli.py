"""
CLI entry points for lipopick.

Installed via ``pip install lipopick[star]``:
    lipopick      — pick particles from micrographs
    lipopick-bin  — bin picks by size and export per-bin STAR files
    lipopick-mpi  — MPI-parallel batch picking (requires ``pip install lipopick[mpi]``)

For interactive / Spyder use, see the editable scripts in ``scripts/``.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


# ======================================================================= #
# Shared argument helpers
# ======================================================================= #

def _add_pick_args(parser: argparse.ArgumentParser) -> None:
    """Add picking-related arguments shared by ``lipopick`` and ``lipopick-mpi``."""
    parser.add_argument("--input", "-i", type=str, nargs="+", required=True,
                        help="Path(s) to micrograph(s) or directories (accepts multiple)")
    parser.add_argument("--outdir", "-o", type=str, required=True,
                        help="Output directory for picks, figures, and JSON")
    parser.add_argument("--dmin", type=float, default=150.0,
                        help="Minimum particle diameter in pixels")
    parser.add_argument("--dmax", type=float, default=500.0,
                        help="Maximum particle diameter in pixels")
    parser.add_argument("--threshold-percentile", type=float, default=99.7,
                        help="Percentile for DoG threshold (per pyramid level)")
    parser.add_argument("--nms-beta", type=float, default=0.8,
                        help="NMS exclusion factor (lower = tighter packing)")
    parser.add_argument("--pixel-size", type=float, default=None,
                        help="Pixel size in Angstroms (figures show nm when set)")
    parser.add_argument("--refine", action="store_true",
                        help="Enable radial edge refinement")
    parser.add_argument("--max-local-contrast", type=float, default=3.0,
                        help="Reject picks with local contrast above this (0=disable)")
    parser.add_argument("--max-overlap", type=float, default=0.3,
                        help="Max circle overlap fraction after refinement (0=disable)")
    parser.add_argument("--n-bins", type=int, default=3,
                        help="Number of size bins in extraction plan")
    parser.add_argument("--star", action="store_true",
                        help="Also write RELION-compatible STAR file")
    parser.add_argument("--overlay", action="store_true",
                        help="Write per-micrograph overlay figures (off by default)")
    parser.add_argument("--histogram", action="store_true",
                        help="Write per-micrograph histogram figures (off by default)")
    parser.add_argument("--show-mic", type=str, default=None,
                        help="Pin a micrograph (name substring) in the summary figure")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Figure DPI")
    parser.add_argument("--log-scale", action="store_true", default=True,
                        help="Log scale on histogram y-axis (default)")
    parser.add_argument("--linear-scale", action="store_true",
                        help="Linear scale on histogram y-axis")
    parser.add_argument("--method", type=str, default="dog",
                        choices=["dog", "template", "combined"],
                        help="Detection method")
    parser.add_argument("--correlation-threshold", type=float, default=0.15,
                        help="NCC threshold for template matching [0, 1]")
    parser.add_argument("--template-radius-step", type=float, default=3.0,
                        help="Pixel step between template radii")
    parser.add_argument("--annulus-width-fraction", type=float, default=0.5,
                        help="Annulus width as fraction of template radius")
    parser.add_argument("--pass2", action="store_true",
                        help="Enable morphological-closing two-pass (CC-based) detection")
    parser.add_argument("--pass2-dmin", type=float, default=None,
                        help="Min diameter for pass-2 DoG (default: 2 * closing_radius)")
    parser.add_argument("--pass2-dmax", type=float, default=None,
                        help="Max diameter for pass-2 DoG (default: 2 * dmax)")
    parser.add_argument("--closing-radius", type=int, default=None,
                        help="Morphological closing SE radius in px (default: dmin)")
    parser.add_argument("--pass2-cc-thresh-frac", type=float, default=0.6,
                        help="DoG threshold = frac * max for CC detection in pass 2 (default: 0.6)")
    parser.add_argument("--pass2-cc-min-dark-frac", type=float, default=0.65,
                        help="Interior darkness gate for pass-2 CC filter (default: 0.65)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Suppress progress output")


def _cfg_from_args(args: argparse.Namespace):
    """Build a PickerConfig from parsed CLI arguments."""
    from lipopick import PickerConfig

    return PickerConfig(
        dmin=args.dmin,
        dmax=args.dmax,
        threshold_percentile=args.threshold_percentile,
        nms_beta=args.nms_beta,
        refine=args.refine,
        max_local_contrast=args.max_local_contrast,
        max_overlap=args.max_overlap,
        pixel_size=args.pixel_size,
        n_size_bins=args.n_bins,
        write_csv=True,
        write_star=args.star,
        write_overlay=args.overlay,
        write_histogram=args.histogram,
        figure_dpi=args.dpi,
        figure_formats=("png", "svg"),
        log_scale=not args.linear_scale,
        detection_method=args.method,
        correlation_threshold=args.correlation_threshold,
        template_radius_step=args.template_radius_step,
        annulus_width_fraction=args.annulus_width_fraction,
        pass2=args.pass2,
        pass2_dmin=args.pass2_dmin,
        pass2_dmax=args.pass2_dmax,
        closing_radius=args.closing_radius,
        pass2_cc_thresh_frac=args.pass2_cc_thresh_frac,
        pass2_cc_min_dark_frac=args.pass2_cc_min_dark_frac,
    )


# ======================================================================= #
# lipopick  —  particle picking
# ======================================================================= #

def pick_main(argv=None):
    """Entry point for ``lipopick`` command."""
    p = argparse.ArgumentParser(
        prog="lipopick",
        description="Multi-scale DoG particle picker for cryo-EM lipoprotein micrographs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_pick_args(p)
    p.add_argument("--workers", "-w", type=int, default=1,
                   help="Number of parallel workers (1 = sequential)")
    args = p.parse_args(argv)

    from lipopick import process_micrograph, process_batch
    from lipopick.io import list_micrographs

    input_paths = [Path(p) for p in args.input]
    outdir = Path(args.outdir)
    cfg = _cfg_from_args(args)

    verbose = not args.quiet
    if verbose:
        method_str = f"  method={args.method}" if args.method != "dog" else ""
        print(f"lipopick  |  dmin={args.dmin}px  dmax={args.dmax}px"
              f"  beta={args.nms_beta}  refine={args.refine}{method_str}")
        for ip in input_paths:
            print(f"  Input:  {ip}")
        print(f"  Output: {outdir}")

    # Single file input
    if len(input_paths) == 1 and input_paths[0].is_file():
        result = process_micrograph(input_paths[0], outdir, cfg=cfg, verbose=verbose)
        results = [result]
    else:
        # Gather micrographs from all input directories/files
        mic_paths = []
        for ip in input_paths:
            if ip.is_dir():
                mic_paths.extend(list_micrographs(ip))
            elif ip.is_file():
                mic_paths.append(ip)
            else:
                print(f"ERROR: Input not found: {ip}", file=sys.stderr)
                sys.exit(1)
        if not mic_paths:
            print("ERROR: No micrographs found in input path(s)", file=sys.stderr)
            sys.exit(1)
        if verbose:
            print(f"  Found {len(mic_paths)} micrograph(s) across {len(input_paths)} input(s)")
        results = process_batch(mic_paths, outdir, cfg=cfg, verbose=verbose,
                                workers=args.workers, show_mic=args.show_mic)

    total = sum(r["n_picks"] for r in results)
    print(f"\nTotal picks: {total}")
    return 0


# ======================================================================= #
# lipopick-mpi  —  MPI-parallel batch picking
# ======================================================================= #

def mpi_pick_main(argv=None):
    """Entry point for ``lipopick-mpi`` command."""
    p = argparse.ArgumentParser(
        prog="lipopick-mpi",
        description="MPI-parallel multi-scale DoG particle picker for cryo-EM micrographs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_pick_args(p)
    args = p.parse_args(argv)

    from lipopick.io import list_micrographs
    from lipopick.mpi import mpi_process_batch

    input_paths = [Path(p) for p in args.input]
    for ip in input_paths:
        if not ip.is_dir():
            print(f"ERROR: --input must be directories for MPI mode (got: {ip})",
                  file=sys.stderr)
            sys.exit(1)

    # Gather micrographs from all input directories
    mic_paths = []
    for ip in input_paths:
        mic_paths.extend(list_micrographs(ip))
    mic_paths = sorted(set(mic_paths))

    if not mic_paths:
        print("ERROR: No micrographs found in input path(s)", file=sys.stderr)
        sys.exit(1)

    outdir = Path(args.outdir)
    cfg = _cfg_from_args(args)
    verbose = not args.quiet

    results = mpi_process_batch(
        mic_paths, outdir, cfg=cfg, verbose=verbose,
        show_mic=args.show_mic,
    )

    # results is non-empty only on rank 0
    if results:
        total = sum(r["n_picks"] for r in results)
        print(f"\nTotal picks: {total}")
    return 0


# ======================================================================= #
# lipopick-bin  —  size binning + STAR export
# ======================================================================= #

def bin_main(argv=None):
    """Entry point for ``lipopick-bin`` command."""
    p = argparse.ArgumentParser(
        prog="lipopick-bin",
        description="Bin particle picks by size and export per-bin RELION STAR files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", "-r", type=str, required=True,
                   help="Directory with pick CSVs and micrograph_qc.csv")
    p.add_argument("--n-bins", type=int, default=6,
                   help="Number of size bins")
    p.add_argument("--bin-mode", default="equal_width",
                   choices=["quantile", "equal_width"],
                   help="Binning strategy")
    p.add_argument("--pixel-size", type=float, required=True,
                   help="Pixel size in Å/px of picking micrographs")
    p.add_argument("--target-pixel-size", type=float, default=None,
                   help="Pixel size in Å/px for extraction (default: same as --pixel-size)")
    p.add_argument("--micrograph-ext", default=".mrc",
                   help="Extension appended to micrograph names in STAR files")
    p.add_argument("--dmin", type=float, default=50.0,
                   help="Min diameter in px (for histogram axis)")
    p.add_argument("--dmax", type=float, default=100.0,
                   help="Max diameter in px (for histogram axis)")
    p.add_argument("--min-bin-count", type=int, default=0,
                   help="Merge tail bins until the last bin has at least this many "
                        "particles (0 = disabled)")
    args = p.parse_args(argv)

    # Lazy imports so --help is fast
    import csv
    import json

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")

    from lipopick.extraction import make_extraction_plan, _next_power_of_2
    from lipopick.io import rescale_picks, write_picks_star

    results_dir = Path(args.results_dir)
    picking_apix = args.pixel_size
    target_apix = args.target_pixel_size if args.target_pixel_size else picking_apix

    # ── Load picks ──────────────────────────────────────────────────────
    qc_path = results_dir / "micrograph_qc.csv"
    if not qc_path.exists():
        print(f"ERROR: QC file not found: {qc_path}", file=sys.stderr)
        sys.exit(1)

    pass_names = []
    with open(qc_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["qc_pass"].strip() == "True":
                pass_names.append(row["micrograph"].strip())
    print(f"Found {len(pass_names)} PASS micrographs")

    all_picks = []
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
    print(f"Loaded {len(all_picks)} total picks")

    if not all_picks:
        print("No picks found — nothing to bin.")
        return 0

    diameters = np.array([p["diameter_px"] for p in all_picks], dtype=np.float32)

    # ── Extraction plan ─────────────────────────────────────────────────
    plan = make_extraction_plan(diameters, n_bins=args.n_bins, bin_mode=args.bin_mode,
                                min_bin_count=args.min_bin_count)
    _print_plan(plan, picking_apix, target_apix)

    # ── Histogram ───────────────────────────────────────────────────────
    from lipopick.viz import save_figure
    import matplotlib.pyplot as plt

    _plot_binned_histogram(
        diameters, plan, results_dir,
        dmin=args.dmin, dmax=args.dmax, pixel_size=picking_apix,
    )

    # ── Assign picks to bins ────────────────────────────────────────────
    bins_picks = {b["bin_id"]: [] for b in plan["bins"]}
    for p in all_picks:
        d = p["diameter_px"]
        for b in plan["bins"]:
            if b["d_lo"] <= d < b["d_hi"]:
                bins_picks[b["bin_id"]].append(p)
                break

    # ── Write per-bin STAR files ────────────────────────────────────────
    px_to_nm = picking_apix / 10.0
    scale = picking_apix / target_apix
    star_dir = results_dir / "star_bins"
    star_dir.mkdir(parents=True, exist_ok=True)

    star_paths = []
    for b in plan["bins"]:
        bid = b["bin_id"]
        picks = bins_picks[bid]
        if not picks:
            continue

        if target_apix != picking_apix:
            picks = rescale_picks(picks, picking_apix, target_apix)

        mic_names = [p["micrograph_name"] + args.micrograph_ext for p in picks]
        d_lo_nm = b["d_lo"] * px_to_nm
        d_hi_nm = b["d_hi"] * px_to_nm
        fname = f"bin{bid}_{d_lo_nm:.1f}-{d_hi_nm:.1f}nm.star"
        star_path = star_dir / fname

        write_picks_star(picks, star_path, pixel_size=target_apix,
                         micrograph_names=mic_names)
        star_paths.append(star_path)

    # ── Extraction summary JSON ─────────────────────────────────────────
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
    summary_path = star_dir / "extraction_summary.json"
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    # ── Summary ─────────────────────────────────────────────────────────
    print(f"\nSTAR files written to: {star_dir}/")
    for sp in star_paths:
        bid = int(sp.stem.split("_")[0].replace("bin", ""))
        print(f"  {sp.name:>30s}  ({len(bins_picks[bid])} picks)")
    print(f"\nExtraction summary: {summary_path}")

    if target_apix != picking_apix:
        print(f"\nCoordinates rescaled: {picking_apix:.4f} -> {target_apix:.4f} Å/px"
              f" (x{scale:.3f})")
    return 0


# ======================================================================= #
# Helpers (shared by bin_main and scripts/size_binning.py)
# ======================================================================= #

def _print_plan(plan: dict, picking_apix: float, target_apix: float) -> None:
    from lipopick.extraction import _next_power_of_2
    px_to_nm = picking_apix / 10.0
    scale = picking_apix / target_apix
    rescaled = (target_apix != picking_apix)

    print(f"\n{'='*80}")
    print(f"  Extraction Plan — {plan['n_particles']} particles in {len(plan['bins'])} bins")
    if rescaled:
        print(f"  Picking: {picking_apix:.4f} Å/px  ->  Target: {target_apix:.4f} Å/px"
              f"  (scale x{scale:.3f})")
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


def _plot_binned_histogram(
    diameters, plan, outdir, dmin, dmax, pixel_size,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from lipopick.viz import save_figure

    px_to_nm = pixel_size / 10.0
    diameters_nm = diameters * px_to_nm
    dmin_nm = dmin * px_to_nm
    dmax_nm = dmax * px_to_nm

    fig, ax = plt.subplots(figsize=(8, 4.5))
    hist_bins = np.linspace(dmin_nm, dmax_nm, 50)
    ax.hist(diameters_nm, bins=hist_bins, color="steelblue", edgecolor="white",
            linewidth=0.5, zorder=2)

    colors = ["#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#3498db", "#e67e22"]
    for i, b in enumerate(plan["bins"]):
        d_lo_nm = b["d_lo"] * px_to_nm
        d_hi_nm = b["d_hi"] * px_to_nm
        color = colors[i % len(colors)]
        ax.axvspan(d_lo_nm, d_hi_nm, alpha=0.10, color=color, zorder=1)
        ax.axvline(d_lo_nm, color=color, linestyle="--", linewidth=1.2, zorder=3)
        ax.text(
            (d_lo_nm + d_hi_nm) / 2, ax.get_ylim()[1] if i == 0 else 0,
            f"Bin {i}\n{b['n_particles']} picks\nbox={b['box_size_px']}px",
            ha="center", va="top", fontsize=7.5, color=color, fontweight="bold",
        )
    last_hi = plan["bins"][-1]["d_hi"] * px_to_nm
    ax.axvline(last_hi, color=colors[(len(plan["bins"])-1) % len(colors)],
               linestyle="--", linewidth=1.2, zorder=3)

    ax.set_xlabel("Diameter (nm)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_xlim(dmin_nm, dmax_nm)
    ax.set_title(
        f"Particle size distribution — {len(diameters)} picks, {len(plan['bins'])} bins",
        fontsize=11,
    )
    fig.tight_layout()
    ymax = ax.get_ylim()[1]
    for txt in ax.texts:
        txt.set_y(ymax * 0.95)

    saved = save_figure(fig, f"size_binning_{len(plan['bins'])}bins", outdir)
    plt.close(fig)
    print(f"Histogram saved to: {saved[0]}")
