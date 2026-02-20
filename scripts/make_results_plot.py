"""
make_results_plot.py — Diagnostic results figure for lipopick.

Generates three synthetic micrographs, runs the picker on each,
and produces a publication-ready summary figure with:
  - Top row: micrographs with detected circles (solid, coloured by
             diameter) and true-size reference circles (dashed white)
  - Bottom row: detected vs true diameter bar chart, NMS demo panel,
                and a combined diameter-accuracy scatter plot.

Usage:
    python scripts/make_results_plot.py

Output:
    outputs/results_plot.png
    outputs/results_plot.svg
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── make repo importable when run as a script ───────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from lipopick import PickerConfig, pick_micrograph

# ── CONFIGURATION ────────────────────────────────────────────────────────────
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DPI  = 300
FACTOR      = 2.0 * 2.0 ** 0.5   # sigma → diameter
# ─────────────────────────────────────────────────────────────────────────────


# ── Synthetic image builder ───────────────────────────────────────────────────

def make_image(size: int, blobs: list, noise_std: float = 0.05, seed: int = 0) -> np.ndarray:
    """
    Return a float32 image with dark Gaussian blobs (cryo-EM convention).

    Each entry in `blobs` is (cx, cy, diameter_px) or (cx, cy, diameter_px, amplitude).
    Amplitude defaults to -1.0 (dark blob). A stronger amplitude (more negative)
    makes the blob score higher in NMS — useful for guaranteeing pick ordering.
    """
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, noise_std, (size, size)).astype(np.float32)
    ys = np.arange(size)[:, None]
    xs = np.arange(size)[None, :]
    for entry in blobs:
        cx, cy, d = entry[0], entry[1], entry[2]
        amp = entry[3] if len(entry) > 3 else -1.0
        sigma = d / FACTOR
        img += amp * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2)).astype(np.float32)
    return img


# ── Three test scenarios ──────────────────────────────────────────────────────

SCENARIOS = [
    {
        "title": "Multi-scale detection\n(3 blobs, 3 size classes)",
        "size": 1024,
        "blobs": [(200, 200, 180), (750, 200, 260), (480, 750, 340)],
        "dmin": 150, "dmax": 400,
        "seed": 42,
    },
    {
        "title": "NMS: two nearby same-size particles",
        "size": 512,
        "blobs": [(110, 256, 160), (390, 256, 160)],
        "dmin": 130, "dmax": 250,
        "seed": 7,
    },
    {
        "title": "Size-aware NMS suppression test\n(inner small absorbed, outer small kept)",
        "size": 700,
        "blobs": [
            (350, 350, 240, -1.5),  # large: r=120, excl≈96px (beta=0.8)
            (380, 310, 160, -1.0),  # small INSIDE (dist=50px) → absorbed into large DoG peak
            (150, 540, 160, -1.0),  # small OUTSIDE (dist=265px) → kept as separate pick
        ],
        "dmin": 140, "dmax": 300,
        "threshold_percentile": 50.0,  # low: large blob's extended DoG inflates high percentiles
        "min_score": 0.02,             # absolute floor above noise
        "seed": 13,
    },
]

LABEL_COLORS = ["#e74c3c", "#2ecc71", "#f39c12", "#9b59b6"]


# ── Run picker on each scenario ───────────────────────────────────────────────

results = []
for sc in SCENARIOS:
    img = make_image(sc["size"], sc["blobs"], seed=sc["seed"])
    cfg = PickerConfig(
        dmin=sc["dmin"], dmax=sc["dmax"],
        threshold_percentile=sc.get("threshold_percentile", 99.0),
        min_score=sc.get("min_score", 0.0),
        write_csv=False, write_overlay=False,
        write_histogram=False, write_extraction_plan=False,
    )
    picks = pick_micrograph(img, cfg)
    results.append({"img": img, "picks": picks, **sc})
    print(f'[{sc["title"].split(chr(10))[0]}] → {picks.shape[0]} picks')


# ── Figure layout ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor("#1a1a2e")      # dark background

# Grid: 2 rows, 4 columns (3 micrograph panels + 1 stats panel, top row;
#                            1 wide stats bar chart spanning bottom)
from matplotlib.gridspec import GridSpec
gs = GridSpec(
    2, 4,
    figure=fig,
    left=0.04, right=0.97,
    top=0.93, bottom=0.08,
    hspace=0.35, wspace=0.25,
    height_ratios=[1.6, 1.0],
)

ax_mics  = [fig.add_subplot(gs[0, i]) for i in range(3)]
ax_stats = fig.add_subplot(gs[0, 3])    # diameter comparison
ax_bar   = fig.add_subplot(gs[1, :])    # bottom wide bar chart

panel_labels = "ABC"


# ── Top row: micrograph panels ────────────────────────────────────────────────

all_true_d, all_det_d, all_det_labels = [], [], []

for ax, res, lbl in zip(ax_mics, results, panel_labels):
    img   = res["img"]
    picks = res["picks"]
    blobs = res["blobs"]
    dmin, dmax = res["dmin"], res["dmax"]

    # Show micrograph (contrast stretch)
    p1, p99 = np.percentile(img, (1, 99))
    ax.imshow(img, cmap="gray", vmin=p1, vmax=p99,
              origin="upper", interpolation="nearest")

    # Colormap for picks
    norm_d = Normalize(vmin=dmin, vmax=dmax)
    cmap   = plt.get_cmap("plasma")

    # Draw TRUE circles (dashed white)
    for i, (cx, cy, d, *_) in enumerate(blobs):
        true_circ = mpatches.Circle(
            (cx, cy), radius=d / 2,
            fill=False, edgecolor="white", linewidth=1.2,
            linestyle="--", alpha=0.65,
        )
        ax.add_patch(true_circ)
        ax.text(cx, cy - d / 2 - 6, f"d={d}px",
                ha="center", va="bottom", fontsize=6.5,
                color="white", alpha=0.75)

    # Draw DETECTED circles (solid, coloured by diameter)
    for pick in picks:
        x, y, sigma, score, ds = pick
        det_d = sigma * FACTOR
        color = cmap(norm_d(det_d))
        circ = mpatches.Circle(
            (x, y), radius=det_d / 2,
            fill=False, edgecolor=color, linewidth=1.8, alpha=0.9,
        )
        ax.add_patch(circ)

    ax.set_xlim(0, img.shape[1])
    ax.set_ylim(img.shape[0], 0)
    ax.set_aspect("equal")
    ax.set_title(res["title"], fontsize=8.5, color="white", pad=4)
    ax.set_xlabel("x (px)", fontsize=7.5, color="#aaaaaa")
    ax.set_ylabel("y (px)" if lbl == "A" else "", fontsize=7.5, color="#aaaaaa")
    ax.tick_params(colors="#888888", labelsize=6.5)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444455")

    # Panel label
    ax.text(0.02, 0.97, lbl, transform=ax.transAxes,
            fontsize=13, fontweight="bold", color="white",
            va="top", ha="left")

    # Collect accuracy data (match picks to truths by nearest centroid)
    for (cx, cy, td, *_) in blobs:
        all_true_d.append(td)
        if picks.shape[0] > 0:
            dists = np.hypot(picks[:, 0] - cx, picks[:, 1] - cy)
            best  = int(np.argmin(dists))
            if dists[best] < td:          # within true radius → matched
                all_det_d.append(picks[best, 2] * FACTOR)
                all_det_labels.append(f"{td}px")
            else:
                all_det_d.append(np.nan)
                all_det_labels.append(f"{td}px (missed)")
        else:
            all_det_d.append(np.nan)
            all_det_labels.append(f"{td}px (missed)")


# ── Top-right: detected vs true diameter scatter ──────────────────────────────

ax_stats.set_facecolor("#0d0d1a")
valid = [(t, d) for t, d in zip(all_true_d, all_det_d) if not np.isnan(d)]
if valid:
    t_vals, d_vals = zip(*valid)
    ax_stats.scatter(t_vals, d_vals, s=60, color="#e056fd", edgecolors="white",
                     linewidths=0.5, zorder=3, label="Detected")

    # 1:1 reference line
    rng_min = min(min(t_vals), min(d_vals)) * 0.85
    rng_max = max(max(t_vals), max(d_vals)) * 1.15
    ax_stats.plot([rng_min, rng_max], [rng_min, rng_max],
                  "--", color="#aaaaaa", linewidth=1, alpha=0.6, label="1:1")

    # ±15% band
    ax_stats.fill_between(
        [rng_min, rng_max],
        [rng_min * 0.85, rng_max * 0.85],
        [rng_min * 1.15, rng_max * 1.15],
        color="#aaaaaa", alpha=0.12, label="±15%",
    )

    # Per-point error labels
    for tv, dv in zip(t_vals, d_vals):
        err = (dv - tv) / tv * 100
        ax_stats.annotate(f"{err:+.0f}%", (tv, dv),
                          textcoords="offset points", xytext=(5, 3),
                          fontsize=6, color="#e0e0e0", alpha=0.85)

    ax_stats.set_xlabel("True diameter (px)", fontsize=8, color="#cccccc")
    ax_stats.set_ylabel("Detected diameter (px)", fontsize=8, color="#cccccc")
    ax_stats.tick_params(colors="#888888", labelsize=7)
    ax_stats.set_xlim(rng_min, rng_max)
    ax_stats.set_ylim(rng_min, rng_max)
    ax_stats.legend(fontsize=6.5, framealpha=0.3,
                    labelcolor="white", facecolor="#0d0d1a")

ax_stats.set_title("Diameter accuracy\n(detected vs. true)", fontsize=8.5,
                   color="white", pad=4)
for spine in ax_stats.spines.values():
    spine.set_edgecolor("#444455")


# ── Bottom: per-blob bar chart of diameter error ──────────────────────────────

ax_bar.set_facecolor("#0d0d1a")

labels, errors, colors_bar = [], [], []
for td, dd, lbl in zip(all_true_d, all_det_d, all_det_labels):
    labels.append(lbl)
    if np.isnan(dd):
        errors.append(np.nan)
        colors_bar.append("#555555")
    else:
        errors.append((dd - td) / td * 100)
        abs_err = abs((dd - td) / td * 100)
        colors_bar.append(
            "#2ecc71" if abs_err < 10 else
            "#f39c12" if abs_err < 20 else "#e74c3c"
        )

x = np.arange(len(labels))
bars = ax_bar.bar(x, [e if not np.isnan(e) else 0 for e in errors],
                  color=colors_bar, edgecolor="#333344", linewidth=0.6,
                  width=0.55, zorder=3)

# ±15% reference lines
ax_bar.axhline(15, color="#f39c12", linewidth=0.9, linestyle="--", alpha=0.7,
               label="±15% threshold")
ax_bar.axhline(-15, color="#f39c12", linewidth=0.9, linestyle="--", alpha=0.7)
ax_bar.axhline(0, color="#aaaaaa", linewidth=0.8, linestyle="-", alpha=0.5)

# Value labels on bars
for xi, (e, c) in enumerate(zip(errors, colors_bar)):
    if not np.isnan(e):
        ax_bar.text(xi, e + (1.5 if e >= 0 else -1.5), f"{e:+.1f}%",
                    ha="center", va="bottom" if e >= 0 else "top",
                    fontsize=7, color="white", fontweight="bold")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels(labels, rotation=25, ha="right",
                        fontsize=7.5, color="#cccccc")
ax_bar.set_ylabel("Diameter error (%)\n(detected − true) / true × 100",
                   fontsize=8, color="#cccccc")
ax_bar.set_title("Per-particle diameter error across all scenarios",
                  fontsize=9, color="white", pad=4)
ax_bar.tick_params(colors="#888888", labelsize=7)
ax_bar.legend(fontsize=7.5, framealpha=0.3, labelcolor="white",
              facecolor="#0d0d1a", loc="upper right")
ax_bar.set_xlim(-0.6, len(labels) - 0.4)

# Grid
ax_bar.yaxis.grid(True, color="#333344", linewidth=0.5, zorder=0)
ax_bar.set_axisbelow(True)
for spine in ax_bar.spines.values():
    spine.set_edgecolor("#444455")

# Legend patches for scenario groupings
n_blobs = [len(sc["blobs"]) for sc in SCENARIOS]
scenario_titles = ["A: Multi-scale", "B: NMS separation", "C: Suppression test"]
boundaries = np.cumsum([0] + n_blobs)
for i, (a, b, t) in enumerate(zip(boundaries[:-1], boundaries[1:], scenario_titles)):
    mid = (a + b - 1) / 2
    ax_bar.axvspan(a - 0.5, b - 0.5, color=LABEL_COLORS[i], alpha=0.07, zorder=0)
    ax_bar.text(mid, ax_bar.get_ylim()[0] * 0.85, t,
                ha="center", va="bottom", fontsize=7,
                color=LABEL_COLORS[i], alpha=0.85, style="italic")

# ── Shared figure annotations ─────────────────────────────────────────────────

fig.suptitle(
    "lipopick — Synthetic Validation Results\n"
    "Dashed circles: true particle outlines  •  Solid circles: detected picks, coloured by diameter",
    fontsize=10, color="white", y=0.98,
)

# ── Save ──────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for fmt in ("png", "svg"):
    out = OUTPUT_DIR / f"results_plot.{fmt}"
    fig.savefig(str(out), dpi=FIGURE_DPI, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"Saved: {out}")

plt.close(fig)
