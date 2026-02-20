# lipopick — Project Context for Claude

## Project Description
Multi-scale DoG (Difference-of-Gaussians) particle picker for cryo-EM micrographs of spherical
lipoproteins (HDL/LDL/VLDL). Detects particles, returning one pick per particle with a per-particle
diameter estimate. No training data required — purely classical.

## Current Status
MPI parallelization deployed and tested on HPC: 300 micrographs in ~110s with 16 ranks
(50,600 picks, ~0.37 s/image). All 42 unit tests pass. Package installable with CLI entry
points (`lipopick`, `lipopick-bin`, `lipopick-mpi`) and per-bin STAR export with coordinate
rescaling.

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
- `lipopick/mpi.py` — MPI parallelization (scatter/gather, 1 rank = 1 core)
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
- [x] ~~Capture large lipid-rich particles~~ — DoG fundamentally cannot detect; accepted as limitation
- [x] Package for distribution (CLI entry points: `lipopick`, `lipopick-bin`)
- [x] Per-bin STAR export with coordinate rescaling (picking → extraction pixel frame)
- [x] MPI parallelization for HPC clusters (`lipopick-mpi` entry point, tested: 300 mics in 110s with 16 ranks)
- [ ] Speed optimization (baseline: ~14.5 s/image single-core, ~0.37 s/image with 16 MPI ranks)
- [ ] Documentation (README, usage examples)

## Known Limitations
- **Large faint particles not detected**: Some LDLp micrographs have large (~120-160px) lipid-rich
  particles that are lighter than typical. The DoG fails to produce local maxima at these locations —
  the faint signal is overwhelmed by nearby dark particles (DoG response ~0.01 vs threshold ~0.02).
  Two-pass detection was attempted but the DoG fundamentally cannot detect these particles; would
  require a different detection method (template matching, mask-then-redetect, or adaptive local
  thresholding).
- Raw DoG scale detection underestimates diameter for hard-edged particles — radial refinement
  compensates but large particles may still be slightly underestimated
- Percentile-based thresholding needs manual tuning per dataset (99.7% for sparse, 80% for dense)
- Dark-fraction filter (replaced old mean-contrast filter) uses fraction of interior pixels below
  image mean > 0.3; works for both dark and lighter particles

## Test Command
```bash
pytest tests/ -v
```

## Real Data Test Command
```bash
python scripts/test_real_data.py
```
