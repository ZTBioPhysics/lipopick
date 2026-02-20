"""
Visualisation helpers: overlay plots, histograms, save_figure.

All figures are saved as PNG (300 DPI) + SVG by default.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless by default; overridden when display available
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


def save_figure(
    fig: plt.Figure,
    name: str,
    outdir: str | Path,
    formats: Sequence[str] = ("png", "svg"),
    dpi: int = 300,
) -> List[Path]:
    """
    Save a matplotlib Figure to one or more formats.

    Parameters
    ----------
    fig : Figure
    name : str
        Base filename without extension.
    outdir : str or Path
        Output directory (created if needed).
    formats : sequence of str
        File formats to write (e.g. ("png", "svg")).
    dpi : int
        Resolution for raster formats.

    Returns
    -------
    list of Path
        Paths to saved files.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        p = outdir / f"{name}.{fmt}"
        fig.savefig(str(p), dpi=dpi, bbox_inches="tight")
        saved.append(p)
    return saved


def _draw_overlay_on_ax(
    ax: plt.Axes,
    image: np.ndarray,
    picks: np.ndarray,
    dmin: float = 150.0,
    dmax: float = 500.0,
    pixel_size: float | None = None,
    cmap: str = "plasma",
    linewidth: float = 1.0,
) -> Tuple[Normalize, plt.cm.ScalarMappable]:
    """
    Draw micrograph + pick circles on an existing Axes.

    Returns (norm, sm) so the caller can add a colorbar if desired.
    """
    factor = 2.0 * 2.0 ** 0.5   # sigma → diameter
    if pixel_size is not None:
        px_to_unit = pixel_size / 10.0
    else:
        px_to_unit = 1.0

    p2, p98 = np.percentile(image, (2, 98))
    ax.imshow(image, cmap="gray", vmin=p2, vmax=p98,
              origin="upper", interpolation="nearest")

    norm = Normalize(vmin=dmin * px_to_unit, vmax=dmax * px_to_unit)
    cm = plt.get_cmap(cmap)

    for pick in picks:
        x, y, sigma = float(pick[0]), float(pick[1]), float(pick[2])
        diameter_px = sigma * factor
        radius = diameter_px / 2.0
        diameter_display = diameter_px * px_to_unit
        color = cm(norm(diameter_display))
        circle = mpatches.Circle(
            (x, y), radius=radius,
            fill=False, edgecolor=color, linewidth=linewidth, alpha=0.85,
        )
        ax.add_patch(circle)

    ax.set_xlim(0, image.shape[1])
    ax.set_ylim(image.shape[0], 0)

    sm = ScalarMappable(cmap=cm, norm=norm)
    sm.set_array([])
    return norm, sm


def plot_picks_overlay(
    image: np.ndarray,
    picks: np.ndarray,
    outdir: str | Path,
    name: str = "picks_overlay",
    dmin: float = 150.0,
    dmax: float = 500.0,
    pixel_size: float | None = None,
    formats: Sequence[str] = ("png", "svg"),
    dpi: int = 300,
    cmap: str = "plasma",
) -> plt.Figure:
    """
    Render micrograph with circles coloured by particle diameter.

    Parameters
    ----------
    image : ndarray, shape (ny, nx)
    picks : ndarray, shape (N, 5)
        Columns: x_full, y_full, sigma_full, score, ds
    outdir : str or Path
    name : str
        Base filename for saved figures.
    dmin, dmax : float
        Diameter range (px) for colormap normalisation.
    pixel_size : float or None
        Pixel size in Å/px.  When set, colorbar shows nm instead of px.
    formats : sequence of str
    dpi : int
    cmap : str

    Returns
    -------
    fig : Figure
    """
    if pixel_size is not None:
        unit = "nm"
    else:
        unit = "px"

    fig, ax = plt.subplots(figsize=(10, 10))
    _, sm = _draw_overlay_on_ax(ax, image, picks, dmin=dmin, dmax=dmax,
                                pixel_size=pixel_size, cmap=cmap)

    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(f"Diameter ({unit})", fontsize=10)

    ax.set_title(f"{name}  —  {len(picks)} picks", fontsize=11)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")

    fig.tight_layout()
    save_figure(fig, name, outdir, formats=formats, dpi=dpi)
    return fig


def plot_size_histogram(
    picks: np.ndarray,
    outdir: str | Path,
    name: str = "size_histogram",
    dmin: float = 150.0,
    dmax: float = 500.0,
    pixel_size: float | None = None,
    extraction_plan: dict | None = None,
    formats: Sequence[str] = ("png", "svg"),
    dpi: int = 300,
) -> plt.Figure:
    """
    Plot diameter distribution histogram, optionally overlaying bin boundaries
    from the extraction plan.

    Parameters
    ----------
    picks : ndarray, shape (N, 5)
    outdir : str or Path
    name : str
    dmin, dmax : float
        Axis limits (px).
    pixel_size : float or None
        Pixel size in Å/px.  When set, x-axis shows nm instead of px.
    extraction_plan : dict or None
        If provided, draw vertical lines at bin boundaries.
    formats : sequence of str
    dpi : int

    Returns
    -------
    fig : Figure
    """
    factor = 2.0 * 2.0 ** 0.5
    diameters_px = picks[:, 2] * factor if len(picks) > 0 else np.array([])

    # Unit conversion
    if pixel_size is not None:
        px_to_unit = pixel_size / 10.0
        unit = "nm"
    else:
        px_to_unit = 1.0
        unit = "px"

    diameters = diameters_px * px_to_unit
    dmin_disp = dmin * px_to_unit
    dmax_disp = dmax * px_to_unit

    fig, ax = plt.subplots(figsize=(7, 4))

    if len(diameters) > 0:
        bins = np.linspace(dmin_disp, dmax_disp, 40)
        ax.hist(diameters, bins=bins, color="steelblue", edgecolor="white", linewidth=0.5)

        # Overlay bin boundaries from extraction plan
        if extraction_plan and "bins" in extraction_plan:
            colors = ["#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]
            for i, b in enumerate(extraction_plan["bins"]):
                d_lo = b["d_lo"] * px_to_unit
                d_hi = b["d_hi"] * px_to_unit
                ax.axvline(d_lo, color=colors[i % len(colors)], linestyle="--",
                           linewidth=1.2,
                           label=f"Bin {i}: {d_lo:.1f}–{d_hi:.1f} {unit} (box={b['box_size_px']})")
            ax.legend(fontsize=8, loc="upper right")

        ax.set_xlabel(f"Diameter ({unit})", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_xlim(dmin_disp, dmax_disp)

    ax.set_title(f"Particle size distribution — {len(diameters)} picks", fontsize=11)
    fig.tight_layout()
    save_figure(fig, name, outdir, formats=formats, dpi=dpi)
    return fig


def plot_batch_summary(
    representative_mics: List[Tuple[str, np.ndarray, np.ndarray]],
    all_picks: np.ndarray,
    n_micrographs: int,
    total_time: float,
    outdir: str | Path,
    name: str = "batch_summary",
    dmin: float = 150.0,
    dmax: float = 500.0,
    pixel_size: float | None = None,
    formats: Sequence[str] = ("png", "svg"),
    dpi: int = 300,
) -> plt.Figure:
    """
    Generate a batch summary figure: 3 representative overlays + histogram.

    Parameters
    ----------
    representative_mics : list of (label, image, picks)
        Three representative micrographs with their picks.
        *label* is displayed as the subplot title (e.g. "Low (95 picks)").
    all_picks : ndarray, shape (N, 5)
        All picks across the batch (columns: x, y, sigma, score, ds).
    n_micrographs : int
        Total number of micrographs processed.
    total_time : float
        Total wall-clock time in seconds.
    outdir, name, dmin, dmax, pixel_size, formats, dpi :
        Same as other plot functions.

    Returns
    -------
    fig : Figure
    """
    factor = 2.0 * 2.0 ** 0.5
    if pixel_size is not None:
        px_to_unit = pixel_size / 10.0
        unit = "nm"
    else:
        px_to_unit = 1.0
        unit = "px"

    outdir = Path(outdir)
    n_rep = len(representative_mics)

    fig = plt.figure(figsize=(5 * max(n_rep, 1), 10))
    gs = gridspec.GridSpec(2, max(n_rep, 1), height_ratios=[1, 0.6], hspace=0.25,
                           wspace=0.05)

    # ── Top row: representative overlays ─────────────────────────────────
    for i, (label, image, picks) in enumerate(representative_mics):
        ax = fig.add_subplot(gs[0, i])
        _draw_overlay_on_ax(ax, image, picks, dmin=dmin, dmax=dmax,
                            pixel_size=pixel_size, linewidth=0.8)
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])

    # ── Bottom row: diameter histogram spanning full width ───────────────
    ax_hist = fig.add_subplot(gs[1, :])

    diameters_px = all_picks[:, 2] * factor if len(all_picks) > 0 else np.array([])
    diameters = diameters_px * px_to_unit
    dmin_disp = dmin * px_to_unit
    dmax_disp = dmax * px_to_unit

    if len(diameters) > 0:
        bins = np.linspace(dmin_disp, dmax_disp, 40)
        ax_hist.hist(diameters, bins=bins, color="steelblue", edgecolor="white",
                     linewidth=0.5)
        ax_hist.set_xlabel(f"Diameter ({unit})", fontsize=11)
        ax_hist.set_ylabel("Count", fontsize=11)
        ax_hist.set_xlim(dmin_disp, dmax_disp)

    # Stats text box
    n_total = len(all_picks)
    picks_per_mic = n_total / max(n_micrographs, 1)
    if len(diameters) > 0:
        d_mean = float(np.mean(diameters))
        d_median = float(np.median(diameters))
        d_min_val = float(np.min(diameters))
        d_max_val = float(np.max(diameters))
        stats_text = (
            f"{n_micrographs} micrographs, {n_total} picks\n"
            f"Mean {picks_per_mic:.0f} picks/mic\n"
            f"Diameter: {d_mean:.1f} mean, {d_median:.1f} median "
            f"({d_min_val:.1f}–{d_max_val:.1f}) {unit}\n"
            f"Time: {total_time:.1f}s ({total_time/max(n_micrographs,1):.2f}s/mic)"
        )
    else:
        stats_text = f"{n_micrographs} micrographs, 0 picks\nTime: {total_time:.1f}s"

    ax_hist.text(
        0.98, 0.95, stats_text, transform=ax_hist.transAxes,
        verticalalignment="top", horizontalalignment="right",
        fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.8),
    )

    ax_hist.set_title(f"Batch summary — {n_total} picks", fontsize=11)
    save_figure(fig, name, outdir, formats=formats, dpi=dpi)
    return fig
