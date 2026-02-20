"""
I/O helpers: read micrographs (MRC / TIFF), write picks (CSV / STAR / JSON).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Optional

import numpy as np


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #

def read_micrograph(path: str | Path) -> np.ndarray:
    """
    Load a micrograph as a 2-D float32 array.

    Supports:
      - MRC / MRC2014  (.mrc, .mrcs)
      - TIFF           (.tif, .tiff)

    Returns array with shape (ny, nx), dtype float32.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".mrc", ".mrcs"}:
        return _read_mrc(path)
    elif suffix in {".tif", ".tiff"}:
        return _read_tiff(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix!r}. Use .mrc or .tif/.tiff")


def _read_mrc(path: Path) -> np.ndarray:
    try:
        import mrcfile
    except ImportError:
        raise ImportError("mrcfile is required to read MRC files: pip install mrcfile")

    with mrcfile.open(str(path), mode="r", permissive=True) as mrc:
        data = mrc.data.astype(np.float32)

    if data.ndim == 3:
        if data.shape[0] == 1:
            data = data[0]
        else:
            raise ValueError(
                f"MRC file has {data.shape[0]} frames; pass a single 2-D micrograph"
            )
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D micrograph, got shape {data.shape}")
    return data


def _read_tiff(path: Path) -> np.ndarray:
    try:
        import tifffile
    except ImportError:
        raise ImportError("tifffile is required to read TIFF files: pip install tifffile")

    data = tifffile.imread(str(path)).astype(np.float32)
    if data.ndim == 3:
        if data.shape[0] == 1:
            data = data[0]
        elif data.shape[2] in {1, 3, 4}:
            # HxWxC — convert to grayscale
            data = data.mean(axis=2)
        else:
            data = data[0]
    if data.ndim != 2:
        raise ValueError(f"Expected 2-D micrograph, got shape {data.shape}")
    return data


# --------------------------------------------------------------------------- #
# Writing — CSV
# --------------------------------------------------------------------------- #

PICKS_CSV_FIELDS = [
    "x_px", "y_px", "diameter_px", "score", "pyramid_level", "sigma_px"
]


def write_picks_csv(picks: List[dict], path: str | Path) -> None:
    """Write picks list to CSV. Each dict must have keys matching PICKS_CSV_FIELDS."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PICKS_CSV_FIELDS)
        writer.writeheader()
        for p in picks:
            writer.writerow({k: p.get(k, "") for k in PICKS_CSV_FIELDS})


def read_picks_csv(path: str | Path) -> List[dict]:
    """Read a picks CSV back into a list of dicts."""
    path = Path(path)
    picks = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            picks.append({
                "x_px": float(row["x_px"]),
                "y_px": float(row["y_px"]),
                "diameter_px": float(row["diameter_px"]),
                "score": float(row["score"]),
                "pyramid_level": int(row["pyramid_level"]),
                "sigma_px": float(row["sigma_px"]),
            })
    return picks


COMBINED_CSV_FIELDS = ["micrograph"] + PICKS_CSV_FIELDS


def write_combined_csv(
    all_picks: List[dict],
    path: str | Path,
) -> None:
    """
    Write a combined CSV with a ``micrograph`` column prepended.

    Each dict in *all_picks* must have the keys in ``PICKS_CSV_FIELDS`` plus
    a ``"micrograph"`` key (stem name, no extension).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COMBINED_CSV_FIELDS)
        writer.writeheader()
        for p in all_picks:
            writer.writerow({k: p.get(k, "") for k in COMBINED_CSV_FIELDS})


# --------------------------------------------------------------------------- #
# Writing — STAR (RELION-compatible)
# --------------------------------------------------------------------------- #

def write_picks_star(
    picks: List[dict],
    path: str | Path,
    pixel_size: float = 1.0,
    micrograph_names: Optional[List[str]] = None,
) -> None:
    """
    Write picks to a RELION 3.1+ two-block STAR file.

    Parameters
    ----------
    picks : list of dict
        Each dict has x_px, y_px, diameter_px, score, etc.
    path : str or Path
        Output .star file path.
    pixel_size : float
        Pixel size in Å/px for the coordinate frame (written to optics table).
    micrograph_names : list of str, optional
        Per-particle micrograph filename. If None, column is omitted.

    Coordinates are 1-indexed (RELION convention).
    Requires: pip install starfile pandas
    """
    try:
        import starfile
        import pandas as pd
    except ImportError:
        raise ImportError(
            "STAR output requires starfile and pandas: pip install starfile pandas"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Optics table (RELION 3.1+)
    optics = pd.DataFrame([{
        "rlnOpticsGroupName": "opticsGroup1",
        "rlnOpticsGroup": 1,
        "rlnImagePixelSize": pixel_size,
    }])

    # Particles table
    rows = []
    for i, p in enumerate(picks):
        row = {
            "rlnCoordinateX": p["x_px"] + 1,   # 0→1 indexed
            "rlnCoordinateY": p["y_px"] + 1,
            "rlnAutopickFigureOfMerit": p["score"],
            "rlnOpticsGroup": 1,
        }
        if micrograph_names is not None:
            row["rlnMicrographName"] = micrograph_names[i]
        rows.append(row)

    particles = pd.DataFrame(rows)
    starfile.write({"optics": optics, "particles": particles}, str(path))


# --------------------------------------------------------------------------- #
# Coordinate rescaling
# --------------------------------------------------------------------------- #

def rescale_picks(
    picks: List[dict], src_apix: float, dst_apix: float,
) -> List[dict]:
    """
    Rescale pick coordinates and sizes from one pixel frame to another.

    Parameters
    ----------
    picks : list of dict
        Picks with x_px, y_px, diameter_px, sigma_px keys.
    src_apix : float
        Source pixel size (Å/px) — the frame the picks are currently in.
    dst_apix : float
        Destination pixel size (Å/px) — the frame to rescale into.

    Returns
    -------
    list of dict
        New pick dicts with rescaled spatial fields. Non-spatial fields
        (score, pyramid_level, etc.) are copied unchanged.
    """
    if src_apix == dst_apix:
        return picks

    scale = src_apix / dst_apix
    rescaled = []
    for p in picks:
        rp = dict(p)
        rp["x_px"] = p["x_px"] * scale
        rp["y_px"] = p["y_px"] * scale
        rp["diameter_px"] = p["diameter_px"] * scale
        if "sigma_px" in p:
            rp["sigma_px"] = p["sigma_px"] * scale
        rescaled.append(rp)
    return rescaled


# --------------------------------------------------------------------------- #
# Writing — Extraction plan JSON
# --------------------------------------------------------------------------- #

def write_extraction_plan(plan: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(plan, fh, indent=2)


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #

def list_micrographs(directory: str | Path, extensions=(".mrc", ".mrcs", ".tif", ".tiff")) -> List[Path]:
    """Return sorted list of micrograph paths in a directory."""
    directory = Path(directory)
    paths = []
    for ext in extensions:
        paths.extend(directory.glob(f"*{ext}"))
    return sorted(set(paths))
