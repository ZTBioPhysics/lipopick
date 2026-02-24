#!/usr/bin/env python3
"""
make_relion_jobs.py — Generate RELION v5 extraction and 2D class SBATCH scripts.

Reads extraction_summary.json produced by lipopick-bin and writes:
  - Per-bin extraction SBATCH scripts  (relion_extract)
  - Per-bin 2D class SBATCH scripts    (relion_refine --class2d)
  - submit_all.sh                      (submits all jobs in dependency order)

Usage:
    python scripts/make_relion_jobs.py \\
        --star-dir /path/to/star_bins/ \\
        --relion-dir /path/to/relion_project/

Spyder / notebook:
    Edit the CONFIGURATION block below and run directly.

Before running:
  1. Fix rlnMicrographName paths in each STAR file to point to raw micrographs
  2. Ensure micrographs are accessible at those paths on the cluster
  3. Adjust SBATCH resource parameters to match available nodes
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

# ============================================================
# CONFIGURATION — Edit these for your run (Spyder-friendly)
# ============================================================
STAR_DIR        = "/path/to/star_bins"      # output of lipopick-bin
RELION_DIR      = "/path/to/relion_project" # RELION project root

# SBATCH resources — adjust to your cluster
ACCOUNT         = "general"
EXTRACT_PARTITION = "general"
CLASS2D_PARTITION = "gpu"              # GPU partition name on your cluster
EXTRACT_CPUS    = 16
CLASS2D_CPUS    = 8
CLASS2D_GPUS    = 1                   # GPUs per 2D class job
EXTRACT_MEM     = "64G"
CLASS2D_MEM     = "32G"
EXTRACT_TIME    = "04:00:00"
CLASS2D_TIME    = "24:00:00"

# 2D class parameters — adjust per dataset as needed
CLASS2D_ITER    = 25
CLASS2D_TAU     = 2                   # tau2_fudge (higher = more aggressive)
MIN_CLASSES     = 20
MAX_CLASSES     = 200
PICKS_PER_CLASS = 300                 # n_particles / PICKS_PER_CLASS = K
# ============================================================


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate RELION v5 extraction and 2D class SBATCH scripts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--star-dir",   type=str, default=None,
                   help="Directory containing lipopick-bin STAR files and extraction_summary.json")
    p.add_argument("--relion-dir", type=str, default=None,
                   help="RELION project root directory")
    p.add_argument("--account",    type=str, default=None)
    p.add_argument("--extract-partition", type=str, default=None)
    p.add_argument("--class2d-partition", type=str, default=None)
    p.add_argument("--extract-cpus", type=int, default=None)
    p.add_argument("--class2d-cpus", type=int, default=None)
    p.add_argument("--class2d-gpus", type=int, default=None)
    return p.parse_args(argv)


def _n_classes(n_particles: int) -> int:
    return min(MAX_CLASSES, max(MIN_CLASSES, n_particles // PICKS_PER_CLASS))


def _mask_diameter_ang(d_hi_nm: float) -> int:
    """Mask diameter in Å: 110% of bin upper edge, rounded to nearest 10."""
    raw = d_hi_nm * 10.0 * 1.1
    return int(round(raw / 10.0) * 10)


def _bg_radius(box_px: int) -> int:
    """Normalization bg_radius: box / 4, rounded down to nearest integer."""
    return box_px // 4


def _find_star(star_dir: Path, bin_id: int) -> Path | None:
    """Find the STAR file for a given bin_id."""
    matches = list(star_dir.glob(f"bin{bin_id}_*.star"))
    return matches[0] if matches else None


def _write_extract_script(
    bin_id: int,
    star_path: Path,
    box_px: int,
    relion_dir: Path,
    scripts_dir: Path,
    account: str,
    partition: str,
    ncpus: int,
) -> Path:
    bg_rad = _bg_radius(box_px)
    extract_out = relion_dir / f"Extract/bin{bin_id}"
    script_path = scripts_dir / f"extract_bin{bin_id}.sh"

    script = f"""#!/bin/bash
#SBATCH --job-name="extract_bin{bin_id}"
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={ncpus}
#SBATCH --mem={EXTRACT_MEM}
#SBATCH --time={EXTRACT_TIME}
#SBATCH --output={scripts_dir}/logs/extract_bin{bin_id}_%j.out
#SBATCH --error={scripts_dir}/logs/extract_bin{bin_id}_%j.err

# ── RELION v5 Extraction — bin {bin_id} ────────────────────────────────────────
# Box size:  {box_px} px at target pixel size
# bg_radius: {bg_rad} px (box/4)
#
# NOTE: Before submitting, ensure rlnMicrographName paths in the STAR file
#       point to the raw (unbinned) micrographs on this cluster.

relion_extract \\
    --i {star_path} \\
    --o {extract_out}/ \\
    --extract_size {box_px} \\
    --invert_contrast \\
    --norm \\
    --bg_radius {bg_rad} \\
    --j {ncpus}

echo "Extraction bin{bin_id} done: $(date)"
"""
    script_path.write_text(script)
    return script_path


def _write_class2d_script(
    bin_id: int,
    n_particles: int,
    d_hi_nm: float,
    box_px: int,
    relion_dir: Path,
    scripts_dir: Path,
    account: str,
    partition: str,
    ncpus: int,
    ngpus: int,
) -> Path:
    k = _n_classes(n_particles)
    mask_ang = _mask_diameter_ang(d_hi_nm)
    extract_particles = relion_dir / f"Extract/bin{bin_id}/particles.star"
    class2d_out = relion_dir / f"Class2D/bin{bin_id}"
    script_path = scripts_dir / f"class2d_bin{bin_id}.sh"

    gpu_line = f"#SBATCH --gres=gpu:{ngpus}" if ngpus > 0 else ""
    gpu_arg  = '--gpu ""' if ngpus > 0 else ""

    script = f"""#!/bin/bash
#SBATCH --job-name="class2d_bin{bin_id}"
#SBATCH --account={account}
#SBATCH --partition={partition}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={ncpus}
#SBATCH --mem={CLASS2D_MEM}
#SBATCH --time={CLASS2D_TIME}
{gpu_line}
#SBATCH --output={scripts_dir}/logs/class2d_bin{bin_id}_%j.out
#SBATCH --error={scripts_dir}/logs/class2d_bin{bin_id}_%j.err

# ── RELION v5 2D Classification — bin {bin_id} ────────────────────────────────
# Particles:       {n_particles:,}
# Classes (K):     {k}
# Mask diameter:   {mask_ang} Å  ({d_hi_nm:.1f} nm × 1.1)
# Box size:        {box_px} px
#
# Adjust K, tau2_fudge, iter, and particle_diameter as needed for your data.

relion_refine \\
    --i {extract_particles} \\
    --o {class2d_out}/ \\
    --class2d \\
    --K {k} \\
    --iter {CLASS2D_ITER} \\
    --tau2_fudge {CLASS2D_TAU} \\
    --particle_diameter {mask_ang} \\
    --flatten_solvent \\
    --zero_mask \\
    --ctf \\
    --norm \\
    --scale \\
    --j {ncpus} \\
    {gpu_arg}

echo "2D class bin{bin_id} done: $(date)"
"""
    script_path.write_text(script)
    return script_path


def _write_submit_all(
    bins: list,
    scripts_dir: Path,
    extract_scripts: dict,
    class2d_scripts: dict,
) -> Path:
    submit_path = scripts_dir / "submit_all.sh"
    lines = [
        "#!/bin/bash",
        "# submit_all.sh — Submit all extraction jobs, then 2D class jobs as dependencies",
        "# Run from the scripts directory: bash submit_all.sh",
        "",
        "declare -A EXTRACT_JIDS",
        "",
        "# ── Extraction jobs ──────────────────────────────────────────────────",
    ]
    for b in bins:
        bid = b["bin_id"]
        lines.append(
            f'EXTRACT_JIDS[{bid}]=$(sbatch --parsable {extract_scripts[bid].name})'
        )
        lines.append(
            f'echo "Submitted extract_bin{bid}: job ${{EXTRACT_JIDS[{bid}]}}"'
        )

    lines += [
        "",
        "# ── 2D class jobs (depend on extraction) ─────────────────────────────",
    ]
    for b in bins:
        bid = b["bin_id"]
        lines.append(
            f'sbatch --dependency=afterok:${{EXTRACT_JIDS[{bid}]}} {class2d_scripts[bid].name}'
        )
        lines.append(f'echo "Submitted class2d_bin{bid} (after extract job ${{EXTRACT_JIDS[{bid}]}})"')

    submit_path.write_text("\n".join(lines) + "\n")
    submit_path.chmod(0o755)
    return submit_path


def main(argv=None):
    args = parse_args(argv)

    star_dir   = Path(args.star_dir)   if args.star_dir   else Path(STAR_DIR)
    relion_dir = Path(args.relion_dir) if args.relion_dir else Path(RELION_DIR)
    account    = args.account              or ACCOUNT
    ex_part    = args.extract_partition    or EXTRACT_PARTITION
    c2d_part   = args.class2d_partition    or CLASS2D_PARTITION
    ex_cpus    = args.extract_cpus         or EXTRACT_CPUS
    c2d_cpus   = args.class2d_cpus         or CLASS2D_CPUS
    c2d_gpus   = args.class2d_gpus         or CLASS2D_GPUS

    # Load extraction plan
    summary_path = star_dir / "extraction_summary.json"
    if not summary_path.exists():
        print(f"ERROR: {summary_path} not found — run lipopick-bin first", file=sys.stderr)
        sys.exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    bins = summary["bins"]
    picking_apix = summary["picking_pixel_size_angstrom"]
    target_apix  = summary["target_pixel_size_angstrom"]

    print(f"Loaded {len(bins)} bins  |  "
          f"picking {picking_apix} Å/px → extraction {target_apix} Å/px")

    # Output directory for SBATCH scripts
    scripts_dir = star_dir / "relion_scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "logs").mkdir(exist_ok=True)

    extract_scripts = {}
    class2d_scripts = {}

    print(f"\n{'Bin':>4}  {'d_lo (nm)':>10}  {'d_hi (nm)':>10}  "
          f"{'n_picks':>8}  {'box (px)':>8}  {'K':>4}  {'mask (Å)':>9}")
    print(f"{'---':>4}  {'-'*10}  {'-'*10}  {'------':>8}  {'------':>8}  {'--':>4}  {'-------':>9}")

    for b in bins:
        bid      = b["bin_id"]
        d_lo_nm  = b["d_lo_nm"]
        d_hi_nm  = b["d_hi_nm"]
        n        = b["n_particles"]
        box_px   = b["box_size_target_px"]
        k        = _n_classes(n)
        mask_ang = _mask_diameter_ang(d_hi_nm)

        star_path = _find_star(star_dir, bid)
        if star_path is None:
            print(f"  WARNING: no STAR file found for bin {bid}, skipping")
            continue

        extract_scripts[bid] = _write_extract_script(
            bid, star_path, box_px, relion_dir, scripts_dir,
            account, ex_part, ex_cpus,
        )
        class2d_scripts[bid] = _write_class2d_script(
            bid, n, d_hi_nm, box_px, relion_dir, scripts_dir,
            account, c2d_part, c2d_cpus, c2d_gpus,
        )
        print(f"{bid:>4}  {d_lo_nm:>10.1f}  {d_hi_nm:>10.1f}  "
              f"{n:>8,}  {box_px:>8}  {k:>4}  {mask_ang:>9}")

    submit_path = _write_submit_all(bins, scripts_dir, extract_scripts, class2d_scripts)

    print(f"\nScripts written to: {scripts_dir}/")
    print(f"  extract_bin{{N}}.sh  — {len(extract_scripts)} extraction jobs")
    print(f"  class2d_bin{{N}}.sh  — {len(class2d_scripts)} 2D class jobs")
    print(f"  submit_all.sh       — submits all with dependencies")
    print(f"\nBefore submitting:")
    print(f"  1. Fix rlnMicrographName paths in each STAR file")
    print(f"  2. Adjust SBATCH resource parameters in each script")
    print(f"  3. cd {scripts_dir} && bash submit_all.sh")


if __name__ == "__main__":
    sys.exit(main())
