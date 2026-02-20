# Project Prompt: Fast Multi-Scale Particle Picker for Variable-Size Spherical Nanoparticles (Cryo-EM)

## Context
I am a cryo-EM scientist working with lipoproteins and other roughly spherical nanoparticles. My datasets often contain a wide range of particle sizes in the same micrographs. Standard blob/template pickers require a single diameter or narrow range; if the diameter is set small, large particles get picked multiple times (peppered by many small blob matches). If set large, small particles are missed. Variable particle sizes also complicate extraction because different sizes ideally need different box sizes.

I can provide **pre-denoised micrographs** (from a neural network denoiser). Typical pixel size is **~1 Å/px**. Typical particle diameters are **15–50 nm** (≈ **150–500 px**).

## Goal
Build a **fast, simple, effective** particle picking tool for **variable-size spherical/near-spherical particles** that:

1. Produces **one pick per particle** (center coordinates).
2. Estimates an **individual particle diameter** per pick.
3. Outputs a **size distribution** (histogram) from picks, so I can decide how many extraction jobs/bins to run.
4. Exports picks in **standard cryo-EM formats** (at minimum CSV; ideally RELION `.star`; optional cryoSPARC-friendly output).

## Preferred Approach (MVP)
Use a classical **multi-scale blob detector** based on **scale-space LoG/DoG** with **local maxima in (x, y, scale)** and **size-aware non-maximum suppression**. This naturally yields both center and size and prevents repeated picks across large particles.

### Why this approach
- No training data required.
- Works well for blob-like spherical particles.
- Provides a direct size estimate per detection.
- Can be made fast using a multi-resolution image pyramid.

## Key Requirements
### Input
- Denoised micrograph(s), e.g. MRC or TIFF (float32). Assume coordinates refer to the same pixel grid as the originals.
- Pixel size (default 1 Å/px).
- Expected diameter range: default **150–500 px** (15–50 nm at 1 Å/px).

### Output
For each pick output a row with:
- `x_px`, `y_px` (particle center, in full-resolution pixels)
- `diameter_px` (estimated)
- `score` (blob response)
- optional: `pyramid_level`, `sigma`

Also output:
- A size distribution summary (histogram data + optional plot).
- Suggested extraction bins and box sizes (see below).

## Algorithm Specification
### 1) Multi-resolution pyramid (for speed)
Use downsample factors `ds ∈ {1, 2, 4}`. Run detection at each level with a sigma range appropriate to that scale, then map detections back to full resolution:
- `x_full = x_level * ds`
- `y_full = y_level * ds`
- `diameter_full = diameter_level * ds`

Suggested diameter coverage per pyramid level (in full-res pixels):
- ds=1: ~150–250 px
- ds=2: ~250–400 px
- ds=4: ~400–500 px

### 2) Scale-space response (DoG or LoG)
Compute a scale-space blob response across sigmas in each pyramid level:
- Use **Difference-of-Gaussians (DoG)** as a fast approximation of LoG.
- Use **geometric spacing** for sigma: `sigma_i = sigma_min * k^i`
  - default `k = 1.10` (optionally 1.08 for finer size resolution)

Use the common LoG blob relation for size conversion:
- `diameter_px ≈ 2 * sqrt(2) * sigma ≈ 2.828 * sigma`
- thus `sigma ≈ diameter_px / 2.828`

### 3) Candidate detection: local maxima in (x, y, sigma)
Build a response stack `R[s, y, x]` and find **3D local maxima** in a 3×3×3 neighborhood over `(scale, y, x)`.

Thresholding:
- Start with a simple robust threshold: e.g. top `99.7` percentile of response values per micrograph/stack.
- Make threshold configurable.

### 4) Merge all candidates and de-duplicate with size-aware NMS
Combine candidates from all pyramid levels (in full-res coordinates), then apply a **size-aware NMS** to ensure only one center per particle and suppress “small picks inside big particles”.

Proposed NMS rule:
- Sort candidates by `score` descending.
- Accept a candidate unless it lies within an exclusion radius of an already-accepted detection.
- Exclusion radius derived from accepted detection size:
  - Let accepted radius `rA = dA/2`.
  - Reject candidate if `distance(candidate, accepted) < beta * rA`.
  - default `beta = 0.8` (configurable; try 0.6 if you expect tightly packed neighboring particles).

### 5) Optional refinement (nice-to-have)
After accepting a pick, refine diameter using a fast radial profile method on the denoised micrograph:
- Compute radial mean or radial gradient magnitude in concentric rings around the center.
- Adjust radius to the strongest edge near the initial estimate (± ~25% window).
This improves size accuracy and rejects obvious false positives.

## Extraction Job Suggestions (Post-pick)
Use the picked `diameter_px` values to propose how many extraction streams/jobs to run.

Implement:
1. Build histogram of diameters (bin width ~10–15 px).
2. Smooth histogram (small Gaussian over bins).
3. Detect peaks (modes) to decide number of bins:
   - If unimodal but broad → default 3 bins (quantiles).
   - If 2–3 clear modes → set bins at valleys between peaks.
   - Cap at 4 bins by default.

For each size bin, propose a box size:
- `box_px = round_up_to_multiple(pad * d_max_in_bin, multiple=32)`
- default `pad = 1.5` (configurable 1.4–1.6)

Example typical outcomes for 150–500 px diameters might be ~384 / 544 / 768 px box sizes (depending on bin maxima).

## Deliverables
1. A Python package or CLI tool, e.g. `lipopick`:
   - `lipopick pick --in micrographs/*.mrc --out picks/ --pixA 1.0 --dmin 150 --dmax 500`
   - Supports batch processing.
2. Outputs:
   - `picks.csv` per micrograph (or combined), plus optional `picks.star` for RELION import.
   - `sizes.csv` summary + histogram plot image (optional).
   - `extraction_plan.json` or `.md` describing suggested size bins and box sizes.
3. Configurable parameters:
   - sigma spacing `k`
   - percentile threshold
   - NMS `beta`
   - pyramid levels
   - padding factor for box size

## Non-goals (for MVP)
- No deep learning training required.
- No interactive GUI required initially.
- Focus on denoised micrographs and spherical-ish particles.

## Notes / Pitfalls
- Ensure coordinate conventions are correct for RELION STAR (RELION often expects 1-indexed coordinates).
- Ensure denoised micrographs preserve the same pixel grid and scaling as originals; otherwise provide a coordinate mapping step.
- Provide runtime-friendly implementation (pyramid + separable Gaussian filtering; avoid extremely large full-res kernels).
