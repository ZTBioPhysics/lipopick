# lipopick — Project Context for Claude

## Project Description
Multi-scale DoG (Difference-of-Gaussians) particle picker for cryo-EM micrographs of spherical
lipoproteins (HDL/LDL/VLDL). Detects particles, returning one pick per particle with a per-particle
diameter estimate. No training data required — purely classical.

## Current Status
CC-based two-pass detection implemented and validated. All 77 unit tests pass. Pass-2 uses
morphological closing + connected-component analysis (not NMS-based) with a dark-fraction
interior filter. Validated on 4 known-TP micrographs: 9/9 TPs detected with 0 false
positives. Logic review of all core algorithm files completed — comments/docstrings corrected,
no behavioral changes. Ready to run on HPC cluster with `--pass2` flag.

## Architecture
- `lipopick/config.py` — PickerConfig dataclass (all tunable parameters)
- `lipopick/io.py` — MRC/TIFF reading; CSV/STAR/JSON writing
- `lipopick/pyramid.py` — Anti-aliased downsampling pyramid (ds=1,2,4)
- `lipopick/dog.py` — DoG scale-space stack + 3D local maxima detection
- `lipopick/nms.py` — Size-aware NMS with KD-tree (key to one-pick-per-particle)
- `lipopick/refine.py` — Radial edge refinement (corrects DoG diameter underestimation)
- `lipopick/extraction.py` — Histogram binning + extraction plan (box sizes)
- `lipopick/viz.py` — Overlay plots, histograms, save_figure helper
- `lipopick/pipeline.py` — Orchestrator: pick_micrograph / process_batch + post-processing filters
- `lipopick/morph.py` — Morphological closing for two-pass detection (`_disk_footprint`, `morphological_clean`)
- `lipopick/mpi.py` — MPI parallelization (scatter/gather, 1 rank = 1 core)
- `lipopick/template.py` — Dark-disc NCC template matching (alternative/supplementary detector)
- `lipopick/cli.py` — Installable CLI entry points (`lipopick`, `lipopick-bin`, `lipopick-mpi`)
- `scripts/pick_particles.py` — Spyder-friendly picking script (editable constants + argparse)
- `scripts/size_binning.py` — Spyder-friendly size binning + per-bin STAR export
- `scripts/test_real_data.py` — 4-phase real data evaluation (triage → pilot → QC → summary)

## Key Algorithm Notes
- Default pyramid levels: ds=1 (150–275 px), ds=2 (225–440 px), ds=4 (360–500 px)
- For LDLp at 3.03 Å/px: single level `((1, 50.0, 150.0),)` with threshold=80%, beta=1.3
- sigma = diameter / 2.828 (= 2√2)
- NMS exclusion radius uses the **accepted** particle's radius, not the candidate's
- Per-level percentile thresholding (default: 99.7th percentile; 80% for dense fields)
- Radial refinement: `argmax(np.abs(grad))` for edge detection (works for dark particles)
- Post-refine diameter filter: clips picks outside [dmin, dmax]
- Pixel-statistics filter: two-stage rejection — (a) dark_fraction < 0.3 (bright interior), (b) dark_fraction < 0.6 AND local_contrast < 0.2 (ambiguous + no local contrast vs annulus)
- Float32 throughout for memory efficiency

## Pipeline Post-Processing Steps (pipeline.py)
1. DoG detection + local maxima per pyramid level
2. Merge candidates across levels
3. Edge exclusion (dmax/2 border)
4. Size-aware NMS
5. Radial edge refinement (optional, with r_min/r_max bounds and max_refine_ratio=2.0)
6. Post-refine diameter filter (discard picks outside [dmin, dmax])
7. Pixel-statistics filter (reject if dark_fraction < 0.3, or if dark_fraction < 0.6 AND local_contrast < 0.2)
8. Anti-cluster filter (reject picks containing ≥2 smaller picks inside them)
9. Overlap filter (greedy score-ordered, max_overlap fraction)
10. Pass-2 morphological closing + CC re-detect (optional, `--pass2`)

## QC Flagging (scripts/test_real_data.py)
Per-micrograph metrics with MAD-based outlier detection (>2.5 MAD = flagged):
- `image_std` — high = thick ice / overlapping particles
- `dynamic_range` (p98−p2) — high = extreme contrast = bad ice
- `pick_density` — low = bad/empty image
- `diameter_iqr` — high = heterogeneous sizes = overlapping particles

## Data Location
Real micrographs: `data/raw/denoised_micrographs/` (62 LDLp micrographs)
`data/` is in .gitignore — never committed.

## Environment
Conda env: `lipopick` (Python 3.11)
Install: `pip install -e .` or `pip install -e ".[star]"` for STAR output or `pip install -e ".[mpi]"` for HPC

## CLI Commands (after `pip install -e ".[star]"`)
```bash
# Pick particles
lipopick -i /path/to/micrographs/ -o /path/to/outputs/ \
    --dmin 50 --dmax 150 --pixel-size 3.0341 \
    --threshold-percentile 80 --nms-beta 1.3 --refine

# Bin by size + export per-bin STAR files (with optional coordinate rescaling)
lipopick-bin -r /path/to/outputs/ --pixel-size 3.0341 --target-pixel-size 0.96

# MPI batch processing (requires: pip install -e ".[mpi]")
mpirun -np 32 lipopick-mpi -i /path/to/micrographs/ -o /path/to/outputs/ \
    --dmin 50 --dmax 150 --pixel-size 3.0341 \
    --threshold-percentile 80 --nms-beta 1.3 --refine
```

## Pending Tasks
- [x] Transfer real micrographs from HPC cluster
- [x] Run picker on real cryo-EM data and tune parameters
- [x] Implement QC flagging for bad micrographs
- [x] Run on full 62-micrograph dataset (10,568 picks, ~14.5 s/image with dmax=150, 51 PASS / 11 FLAGGED)
- [x] Validate QC flagging on full dataset
- [x] Package for distribution (CLI entry points: `lipopick`, `lipopick-bin`)
- [x] Per-bin STAR export with coordinate rescaling (picking → extraction pixel frame)
- [x] MPI parallelization for HPC clusters (`lipopick-mpi` entry point, tested: 300 mics in 110s with 16 ranks)
- [x] Multi-input directory support (allow multiple `-i` paths — denoised micrographs are spread across jobs)
- [x] Template matching detector (`--method template`) — works but equivalent to DoG
- [x] Two-pass CC-based detection (`--pass2`) — validated on 4 known-TP micrographs: 9/9 TPs found
- [x] Logic review of core algorithms — stale comments/docstrings corrected
- [ ] Run `--pass2` on full HPC dataset
- [ ] Documentation (README, usage examples)

## Two-Pass Morphological Closing Detection (`--pass2`)

Pass-2 detects large faint particles that pass-1 DoG misses, using connected-component analysis:
1. `grey_closing` with disk SE (radius = `closing_radius`, default = `dmin`) removes dark
   features smaller than the SE — small particles disappear, large ones survive
2. DoG on closed image at ds=2, covering `pass2_dmin`–`pass2_dmax`
3. Max-project DoG stack + Gaussian lowpass (σ=3.0 at ds=2) → smooth response map
4. Threshold at `pass2_cc_thresh_frac` × max → binary image; label connected components
5. Per-CC filters: area ≥ 20 px, diameter ≥ 50 px full-res, center ≥ 30 px from edge,
   interior darkness on **original** image: `dark_frac ≥ pass2_cc_min_dark_frac` (default 0.65)
6. Merge pass-1 + pass-2 picks, deduplicate with NMS

**Important**: NMS merge can cause `len(picks_full) < len(picks1)` because large CC picks
suppress nearby small pass-1 picks. The right check is whether TPs appear in output, not
net count.

Recommended LDLp pass-2 CLI flags:
```bash
lipopick ... --pass2 --pass2-dmax 300
```
Defaults: `closing_radius = dmin`, `pass2_dmin = 2 × closing_radius`, `pass2_dmax = 2 × dmax`,
`pass2_cc_thresh_frac = 0.6`, `pass2_cc_min_dark_frac = 0.65`.

## Known Limitations
- Raw DoG scale detection underestimates diameter for hard-edged particles — radial refinement
  compensates but large particles may still be slightly underestimated
- Percentile-based thresholding needs manual tuning per dataset (99.7% for sparse, 80% for dense)
- Dark-fraction filter uses fraction of interior pixels below image mean > 0.3; works for both
  dark and lighter particles
- Pass-2 CC filter uses `dark_frac ≥ 0.65` on the original image; faint particles embedded in
  the densest regions may still be missed if their interior dark_frac falls below this threshold

## HPC Cluster Deployment (Hellbender)

After pulling new code to the cluster, reinstall the package:
```bash
cd /path/to/cluster/lipopick   # adjust to your cluster path
module load miniconda3/4.10.3_gcc_12.3.0
source activate lipopick
git pull origin main
pip install -e .
```

**Important**: Do NOT run `pip install` without first loading the miniconda module and
activating the conda env — the cluster system Python is 3.6 and will fail.

Then edit `run_mpi.sh` to add `--pass2 --pass2-dmax 300` to the `mpirun` command:
```bash
mpirun -np 16 lipopick-mpi -i /path/to/micrographs/ -o /path/to/outputs/ \
    --dmin 50 --dmax 150 --pixel-size 3.0341 \
    --threshold-percentile 80 --nms-beta 1.3 --refine \
    --pass2 --pass2-dmax 300
```

Then submit:
```bash
sbatch run_mpi.sh
```

## Test Command
```bash
pytest tests/ -v
```

## Real Data Test Command
```bash
python scripts/test_real_data.py
```
