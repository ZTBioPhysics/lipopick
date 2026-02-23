"""
Morphological cleaning: remove small dark features using grey closing.

Grey closing = erosion(dilation(f, B), B) fills dark features narrower than
the structuring element while preserving larger structures.  This is a purely
geometric size filter — contrast doesn't matter.

Used by the two-pass pipeline: pass-1 detects normally on the original image,
then morphological_clean() removes small features so pass-2 can detect large
faint particles that were previously masked by smaller neighbours.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import grey_closing


def _disk_footprint(radius: int) -> np.ndarray:
    """
    Create a circular binary footprint for morphological operations.

    Parameters
    ----------
    radius : int
        Radius of the disk in pixels.  The footprint has shape
        ``(2*radius+1, 2*radius+1)``.

    Returns
    -------
    footprint : ndarray of uint8, shape (2*radius+1, 2*radius+1)
    """
    y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
    return (x * x + y * y <= radius * radius).astype(np.uint8)


def morphological_clean(image: np.ndarray, se_radius: int) -> np.ndarray:
    """
    Remove dark features smaller than ``2 * se_radius`` using grey closing.

    The closing operation fills dark regions narrower than the structuring
    element (SE) from the boundary inward, leaving larger dark features
    intact.  The result is a smooth image suitable for re-detection of
    large particles that were previously hidden by smaller neighbours.

    Parameters
    ----------
    image : ndarray, shape (ny, nx), float32
        Input micrograph.  Not modified.
    se_radius : int
        Radius of the disk structuring element in pixels.
        Features with diameter < ``2 * se_radius`` will be removed.

    Returns
    -------
    cleaned : ndarray, shape (ny, nx), float32
        Closed image with small dark features removed.
    """
    footprint = _disk_footprint(se_radius)
    return grey_closing(image, footprint=footprint).astype(np.float32)
