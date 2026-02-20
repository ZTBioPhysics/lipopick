#!/usr/bin/env python3
"""
pick_particles.py — CLI entry point for lipopick.

Usage (CLI):
    python scripts/pick_particles.py --input data/raw/denoised_micrographs/ \\
        --outdir outputs/ --dmin 150 --dmax 500 --refine

Usage (Spyder / notebook):
    Edit the CONFIGURATION block below and run the script directly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ============================================================
# CONFIGURATION — Edit these for your data (Spyder-friendly)
# ============================================================
INPUT_PATH = "/path/to/micrograph_or_directory"   # file or directory
OUTPUT_DIR = "/path/to/outputs"

DMIN = 150.0      # minimum particle diameter (px)
DMAX = 500.0      # maximum particle diameter (px)
PIXEL_SIZE = None # Å/px; set to show nm in figures (e.g. 3.0341)
THRESHOLD_PERCENTILE = 99.7
NMS_BETA = 0.8
REFINE = False    # enable radial edge refinement
MAX_LOCAL_CONTRAST = 3.0  # reject isolated contaminants (0=disable)
MAX_OVERLAP = 0.3         # max circle overlap after refinement (0=disable)
N_SIZE_BINS = 3   # number of extraction size bins

WRITE_CSV = True
WRITE_STAR = False
WRITE_OVERLAY = True
WRITE_HISTOGRAM = True
FIGURE_DPI = 300
FIGURE_FORMATS = ("png", "svg")
VERBOSE = True
# ============================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="lipopick: multi-scale DoG particle picker for cryo-EM lipoprotein micrographs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", type=str, default=None,
                   help="Path to a single micrograph (.mrc/.tif) or a directory of micrographs")
    p.add_argument("--outdir", "-o", type=str, default=None,
                   help="Output directory for picks, figures, and JSON")
    p.add_argument("--dmin", type=float, default=None,
                   help="Minimum particle diameter in pixels")
    p.add_argument("--dmax", type=float, default=None,
                   help="Maximum particle diameter in pixels")
    p.add_argument("--threshold-percentile", type=float, default=None,
                   help="Percentile for DoG threshold (per pyramid level)")
    p.add_argument("--nms-beta", type=float, default=None,
                   help="NMS exclusion factor (lower = tighter packing)")
    p.add_argument("--pixel-size", type=float, default=None,
                   help="Pixel size in Angstroms (figures show nm when set)")
    p.add_argument("--refine", action="store_true", default=None,
                   help="Enable radial edge refinement")
    p.add_argument("--max-local-contrast", type=float, default=None,
                   help="Reject picks with local contrast above this (0=disable)")
    p.add_argument("--max-overlap", type=float, default=None,
                   help="Max circle overlap fraction after refinement (0=disable)")
    p.add_argument("--n-bins", type=int, default=None,
                   help="Number of size bins in extraction plan")
    p.add_argument("--star", action="store_true", default=None,
                   help="Also write RELION-compatible STAR file")
    p.add_argument("--no-overlay", action="store_true",
                   help="Skip overlay figure")
    p.add_argument("--no-histogram", action="store_true",
                   help="Skip histogram figure")
    p.add_argument("--dpi", type=int, default=None,
                   help="Figure DPI")
    p.add_argument("--quiet", "-q", action="store_true",
                   help="Suppress progress output")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Resolve parameters: CLI args override editable constants
    input_path = Path(args.input) if args.input else Path(INPUT_PATH)
    outdir = Path(args.outdir) if args.outdir else Path(OUTPUT_DIR)
    dmin = args.dmin if args.dmin is not None else DMIN
    dmax = args.dmax if args.dmax is not None else DMAX
    threshold = args.threshold_percentile if args.threshold_percentile is not None else THRESHOLD_PERCENTILE
    beta = args.nms_beta if args.nms_beta is not None else NMS_BETA
    pixel_size = args.pixel_size if args.pixel_size is not None else PIXEL_SIZE
    refine = args.refine if args.refine is not None else REFINE
    max_lc = args.max_local_contrast if args.max_local_contrast is not None else MAX_LOCAL_CONTRAST
    max_ov = args.max_overlap if args.max_overlap is not None else MAX_OVERLAP
    n_bins = args.n_bins if args.n_bins is not None else N_SIZE_BINS
    write_star = args.star if args.star is not None else WRITE_STAR
    write_overlay = not args.no_overlay if args.no_overlay else WRITE_OVERLAY
    write_hist = not args.no_histogram if args.no_histogram else WRITE_HISTOGRAM
    dpi = args.dpi if args.dpi is not None else FIGURE_DPI
    verbose = not args.quiet

    # Lazy import (allows editing constants without import errors)
    try:
        from lipopick import PickerConfig, process_micrograph, process_batch
    except ImportError:
        # Try adding parent directory to path (script run from repo root)
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from lipopick import PickerConfig, process_micrograph, process_batch

    cfg = PickerConfig(
        dmin=dmin,
        dmax=dmax,
        threshold_percentile=threshold,
        nms_beta=beta,
        refine=refine,
        max_local_contrast=max_lc,
        max_overlap=max_ov,
        pixel_size=pixel_size,
        n_size_bins=n_bins,
        write_csv=WRITE_CSV,
        write_star=write_star,
        write_overlay=write_overlay,
        write_histogram=write_hist,
        figure_dpi=dpi,
        figure_formats=tuple(FIGURE_FORMATS),
    )

    if verbose:
        print(f"lipopick  |  dmin={dmin}px  dmax={dmax}px  beta={beta}  refine={refine}")
        print(f"  Input:  {input_path}")
        print(f"  Output: {outdir}")

    if input_path.is_dir():
        results = process_batch(input_path, outdir, cfg=cfg, verbose=verbose)
    elif input_path.is_file():
        result = process_micrograph(input_path, outdir, cfg=cfg, verbose=verbose)
        results = [result]
    else:
        print(f"ERROR: Input not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    total = sum(r["n_picks"] for r in results)
    print(f"\nTotal picks: {total}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
