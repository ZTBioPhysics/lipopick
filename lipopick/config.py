"""
PickerConfig — all tunable parameters for lipopick in one dataclass.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple


def _auto_pyramid_levels(
    dmin: float, dmax: float,
) -> Tuple[Tuple[int, float, float], ...]:
    """
    Auto-compute pyramid levels from the particle diameter range.

    Strategy: each level covers a diameter range whose ratio is at most ~2×
    (so the DoG sigma range stays manageable).  Higher downsample factors are
    used for larger diameters to keep the convolution kernel small.

    Level boundaries (full-res px):
      ds=1 :  up to ~275 px
      ds=2 :  up to ~440 px
      ds=4 :  up to ~700 px
      ds=8 :  above 700 px

    Adjacent levels overlap by ~10% so particles at boundaries are caught by
    at least one level.
    """
    # (ds, max_dhi_for_this_ds)
    ds_tiers = [
        (1, 275.0),
        (2, 440.0),
        (4, 700.0),
        (8, float("inf")),
    ]
    overlap = 0.10  # 10% overlap between adjacent levels

    levels = []
    d_lo = dmin

    for ds, tier_cap in ds_tiers:
        if d_lo >= dmax:
            break
        d_hi = min(dmax, tier_cap)
        levels.append((ds, d_lo, d_hi))
        if d_hi >= dmax:
            break
        # Next level starts with overlap
        d_lo = d_hi * (1.0 - overlap)

    return tuple(levels)


@dataclass
class PickerConfig:
    # ------------------------------------------------------------------ #
    # Particle size range (full-resolution pixels)
    # ------------------------------------------------------------------ #
    dmin: float = 150.0   # minimum particle diameter (px)
    dmax: float = 500.0   # maximum particle diameter (px)

    # ------------------------------------------------------------------ #
    # Pyramid levels
    # Each entry: (downsample_factor, d_lo, d_hi)
    # d_lo / d_hi are full-res pixel diameters covered by this level.
    # 10% overlap at boundaries ensures boundary-size particles are caught.
    # When None (default), levels are auto-computed from dmin/dmax.
    # ------------------------------------------------------------------ #
    pyramid_levels: Optional[Tuple[Tuple[int, float, float], ...]] = None

    # ------------------------------------------------------------------ #
    # DoG scale-space
    # ------------------------------------------------------------------ #
    dog_k: float = 1.10        # geometric step between adjacent sigmas
    dog_min_steps: int = 3     # minimum number of sigma steps per level

    # ------------------------------------------------------------------ #
    # Detection
    # ------------------------------------------------------------------ #
    threshold_percentile: float = 99.7   # per-level DoG threshold
    min_score: float = 0.0               # absolute floor (post-percentile)

    # ------------------------------------------------------------------ #
    # NMS
    # ------------------------------------------------------------------ #
    nms_beta: float = 0.8    # exclusion = beta * accepted_radius
                              # lower (0.6) for tightly packed particles

    # ------------------------------------------------------------------ #
    # Edge exclusion
    # ------------------------------------------------------------------ #
    edge_fraction: float = 0.5   # exclude picks within (dmax/2 * edge_fraction) of edge

    # ------------------------------------------------------------------ #
    # Radial refinement (optional)
    # ------------------------------------------------------------------ #
    refine: bool = False          # enable radial edge refinement
    refine_margin: float = 0.25   # search within ±25% of initial radius

    # ------------------------------------------------------------------ #
    # Pixel size (for nm display in figures)
    # ------------------------------------------------------------------ #
    pixel_size: Optional[float] = None   # Å/px; None → figures use px units

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    write_csv: bool = True
    write_star: bool = False
    write_extraction_plan: bool = True
    write_overlay: bool = False
    write_histogram: bool = False
    figure_dpi: int = 300
    figure_formats: Tuple[str, ...] = ("png", "svg")
    log_scale: bool = True

    # ------------------------------------------------------------------ #
    # Post-refinement filters
    # ------------------------------------------------------------------ #
    max_local_contrast: float = 2.0   # reject picks above this (0 = disable)
    max_overlap: float = 0.3          # max circle overlap fraction (0 = disable)
    overlap_mode: str = "smaller"     # "smaller", "candidate", or "larger"

    # ------------------------------------------------------------------ #
    # Extraction plan
    # ------------------------------------------------------------------ #
    n_size_bins: int = 3    # number of extraction size bins

    # ------------------------------------------------------------------ #
    # Detection method
    # ------------------------------------------------------------------ #
    detection_method: str = "dog"  # "dog", "template", or "combined"

    # ------------------------------------------------------------------ #
    # Template matching parameters
    # ------------------------------------------------------------------ #
    template_radius_step: float = 3.0       # px step between template radii
    correlation_threshold: float = 0.15     # absolute NCC threshold [0, 1]
    annulus_width_fraction: float = 0.5     # annulus width = fraction * radius
    max_annulus_width: float = 40.0         # cap annulus width (px)
    template_min_separation: int = 5        # local maxima neighborhood size

    # ------------------------------------------------------------------ #
    # Pass 2 (morphological closing + CC-based re-detect)
    # ------------------------------------------------------------------ #
    pass2: bool = False                        # enable two-pass detection
    pass2_dmin: Optional[float] = None         # default: 2 * closing_radius
    pass2_dmax: Optional[float] = None         # default: 2.0 * dmax
    closing_radius: Optional[int] = None       # SE radius in px (default: dmin)
    pass2_cc_thresh_frac: float = 0.6          # DoG threshold = frac * max for CC detection
    pass2_cc_min_dark_frac: float = 0.65       # interior darkness gate (fraction < image mean)

    def __post_init__(self):
        if self.dmin <= 0 or self.dmax <= 0:
            raise ValueError("dmin and dmax must be positive")
        if self.dmin >= self.dmax:
            raise ValueError("dmin must be less than dmax")
        if not (0 < self.threshold_percentile < 100):
            raise ValueError("threshold_percentile must be in (0, 100)")
        if not (0 < self.nms_beta <= 1.5):
            raise ValueError("nms_beta should be in (0, 1.5]")
        if self.detection_method != "template" and self.dog_k <= 1.0:
            raise ValueError("dog_k must be > 1.0")
        if self.max_local_contrast < 0:
            raise ValueError("max_local_contrast must be >= 0")
        if self.max_overlap < 0 or self.max_overlap > 1.0:
            raise ValueError("max_overlap must be in [0, 1.0]")
        if self.overlap_mode not in ("smaller", "candidate", "larger"):
            raise ValueError("overlap_mode must be 'smaller', 'candidate', or 'larger'")
        if self.detection_method not in ("dog", "template", "combined"):
            raise ValueError("detection_method must be 'dog', 'template', or 'combined'")
        if not (0 <= self.correlation_threshold <= 1):
            raise ValueError("correlation_threshold must be in [0, 1]")
        if self.template_radius_step <= 0:
            raise ValueError("template_radius_step must be > 0")

        # Pass-2 defaults and validation
        if self.pass2:
            if self.closing_radius is None:
                self.closing_radius = int(self.dmin)
            if self.pass2_dmin is None:
                self.pass2_dmin = float(2 * self.closing_radius)
            if self.pass2_dmax is None:
                self.pass2_dmax = 2.0 * self.dmax
            if self.pass2_dmin >= self.pass2_dmax:
                raise ValueError("pass2_dmin must be less than pass2_dmax")
            if not (0 < self.pass2_cc_thresh_frac < 1):
                raise ValueError("pass2_cc_thresh_frac must be in (0, 1)")
            if not (0 < self.pass2_cc_min_dark_frac <= 1):
                raise ValueError("pass2_cc_min_dark_frac must be in (0, 1]")

        if self.pyramid_levels is None:
            self.pyramid_levels = _auto_pyramid_levels(self.dmin, self.dmax)

    @property
    def sigma_from_diameter(self):
        """Return the conversion factor: sigma = diameter / FACTOR."""
        return 2.0 * 2.0 ** 0.5  # 2√2 ≈ 2.828
