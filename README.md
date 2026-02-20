# lipopick

Multi-scale Difference-of-Gaussians (DoG) particle picker for cryo-EM micrographs of spherical
lipoproteins (HDL, LDL, VLDL). Detects particles 150–500 px in diameter, returning **one pick
per particle** with a per-particle diameter estimate. No training data required — purely
classical, deterministic, and fast.

## Why lipopick?

Standard cryo-EM pickers either miss small particles or fire multiple times on large ones.
lipopick solves this with a three-level image pyramid + **size-aware NMS**: the exclusion
radius during non-maximum suppression is derived from the *accepted* particle's radius, not
the candidate's. A small spurious detection inside a large particle is always suppressed.

## Algorithm

```
Micrograph
  └─ Pyramid (ds=1,2,4)
       └─ DoG scale-space per level
            └─ 3-D local maxima detection
  └─ Merge candidates (full-resolution coordinates)
  └─ Edge exclusion
  └─ Size-aware NMS (KD-tree, sorted by score)
  └─ Optional radial refinement
  └─ Outputs (CSV / STAR / JSON / PNG / SVG)
```

Cryo-EM convention: particles are **dark** (high electron scattering). The raw DoG
`G(bigger) - G(smaller)` gives a positive response at dark blob centers and peaks at
σ ≈ R/√2 (where R is the particle radius), correctly recovering the diameter.

## Installation

```bash
# Create conda environment (recommended)
conda create -n lipopick python=3.11 -y
conda activate lipopick

# Clone and install
git clone https://github.com/ZTBioPhysics/lipopick.git
cd lipopick
pip install -e .

# For STAR file output (RELION-compatible):
pip install -e ".[star]"

# For MPI parallelization on HPC clusters:
pip install -e ".[mpi]"
```

## Quick Start

### Python API

```python
from lipopick import PickerConfig, pick_micrograph, process_micrograph
from lipopick.io import read_micrograph

# Load and pick
image = read_micrograph("micrograph.mrc")
cfg = PickerConfig(dmin=150, dmax=500, refine=True)
picks = pick_micrograph(image, cfg)    # ndarray (N, 5)
# Columns: x_px, y_px, sigma_px, score, ds_factor

# Full pipeline (CSV + overlay + histogram + extraction plan)
result = process_micrograph("micrograph.mrc", outdir="outputs/", cfg=cfg)
print(f"Found {result['n_picks']} particles")
```

### Command-Line

```bash
# Single micrograph
python scripts/pick_particles.py --input micrograph.mrc --outdir outputs/

# Batch (directory of MRC/TIF files)
python scripts/pick_particles.py --input data/raw/ --outdir outputs/ \
    --dmin 150 --dmax 500 --refine --star
```

For Spyder/Jupyter: edit the `CONFIGURATION` block at the top of `scripts/pick_particles.py`
and run the file directly (no CLI args needed).

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dmin` | 150 | Minimum particle diameter (px) |
| `dmax` | 500 | Maximum particle diameter (px) |
| `threshold_percentile` | 99.7 | Per-level DoG threshold (% of positive values) |
| `nms_beta` | 0.8 | NMS exclusion factor (`beta * r_accepted`) |
| `refine` | False | Enable radial edge refinement |
| `n_size_bins` | 3 | Extraction size bins in plan JSON |

For tightly packed particles, lower `nms_beta` to 0.6. For noisy data, lower
`threshold_percentile` to 99.0–99.5.

## Outputs

| File | Description |
|------|-------------|
| `*_picks.csv` | x_px, y_px, diameter_px, score, pyramid_level, sigma_px |
| `*_picks.star` | RELION-compatible coordinates (1-indexed, requires `--star`) |
| `*_extraction_plan.json` | Size bins + recommended box sizes (powers of 2) |
| `*_picks_overlay.png/svg` | Micrograph with circles coloured by diameter |
| `*_size_histogram.png/svg` | Diameter distribution with bin boundaries |

## Diameter Estimation

The detected `sigma_px` satisfies `diameter = sigma * 2√2`. This estimate comes from the
scale at which the raw DoG peaks, which is σ ≈ R/√2 for disk-shaped particles. Enable
`refine=True` for a more accurate per-particle estimate via radial intensity profiling.

## Tests

```bash
pytest tests/ -v
```

All 42 tests pass on synthetic micrographs with known blob positions and sizes.

## Project Structure

```
lipopick/
├── config.py          # PickerConfig dataclass
├── io.py              # MRC/TIFF reading, CSV/STAR/JSON writing
├── pyramid.py         # Anti-aliased downsampling pyramid
├── dog.py             # DoG scale-space + 3-D local maxima
├── nms.py             # Size-aware NMS with KD-tree
├── refine.py          # Radial edge refinement
├── extraction.py      # Extraction plan (size bins + box sizes)
├── viz.py             # Overlay plots, histograms
├── pipeline.py        # Orchestrator
├── mpi.py             # MPI parallelization (scatter/gather)
└── cli.py             # CLI entry points
scripts/
├── pick_particles.py  # Spyder-friendly picking (editable constants + argparse)
├── size_binning.py    # Size binning + per-bin STAR export
└── run_mpi.sh         # Example SLURM job script
tests/
├── conftest.py        # Synthetic fixtures
├── test_dog.py
├── test_nms.py
├── test_pyramid.py
├── test_pipeline.py
└── test_mpi.py
```

## Real Data

Transfer micrographs to `data/raw/denoised_micrographs/` (directory is in `.gitignore`).

```bash
# Example: rsync from HPC
rsync -avz hpc:/path/to/denoised/*.mrc data/raw/denoised_micrographs/

# Then run:
python scripts/pick_particles.py \
    --input data/raw/denoised_micrographs/ \
    --outdir outputs/ \
    --dmin 150 --dmax 500
```

Inspect `outputs/*_picks_overlay.png` to verify pick quality. Adjust `threshold_percentile`
and `nms_beta` as needed for your data.
