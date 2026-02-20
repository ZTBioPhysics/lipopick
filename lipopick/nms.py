"""
Size-aware Non-Maximum Suppression (NMS).

The key insight: suppression radius is derived from the *accepted* particle's
radius, not the candidate's. This prevents small spurious detections inside
large particles from surviving — the big particle's exclusion zone covers them.

Algorithm:
  1. Sort candidates by score descending.
  2. For each candidate (in order):
       - If within beta * r_accepted of any already-accepted pick → reject.
       - Otherwise → accept and add to KD-tree.

KD-tree is rebuilt lazily (in batches) for efficiency.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.spatial import cKDTree


# Candidate array column indices
_X = 0
_Y = 1
_SIGMA = 2
_SCORE = 3
_DS = 4


def size_aware_nms(
    candidates: np.ndarray,
    beta: float = 0.8,
    edge_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Apply size-aware NMS to a candidate array.

    Parameters
    ----------
    candidates : ndarray, shape (N, 5), float32
        Columns: x_full, y_full, sigma_full, score, ds
        (output of dog.find_local_maxima, concatenated across levels)
    beta : float
        Exclusion factor. A candidate is suppressed if it falls within
        beta * r_accepted of any accepted pick. Typical: 0.8; for tight
        packing use 0.6.
    edge_mask : ndarray of bool, shape (N,), optional
        If provided, candidates where edge_mask[i] == True are excluded
        before NMS (pre-filtered by pipeline).

    Returns
    -------
    accepted : ndarray, shape (M, 5), float32
        Surviving candidates in the same column format, sorted by score desc.
    """
    if candidates.shape[0] == 0:
        return candidates

    cands = candidates.copy()

    # Apply edge mask
    if edge_mask is not None:
        cands = cands[~edge_mask]
        if cands.shape[0] == 0:
            return cands

    # Sort by score descending
    order = np.argsort(cands[:, _SCORE])[::-1]
    cands = cands[order]

    accepted_xy: List[np.ndarray] = []   # 2-D coords of accepted picks
    accepted_r: List[float] = []          # exclusion radius of accepted picks
    accepted_idx: List[int] = []          # indices into sorted cands

    # KD-tree rebuilt periodically
    tree: cKDTree | None = None
    tree_dirty = True

    for i, cand in enumerate(cands):
        x, y, sigma = cand[_X], cand[_Y], cand[_SIGMA]
        r_cand = sigma * 2.0 * 2.0 ** 0.5 / 2.0   # sigma → radius

        if len(accepted_xy) == 0:
            # First pick always accepted
            _accept(accepted_xy, accepted_r, accepted_idx, x, y, r_cand, i)
            tree_dirty = True
            continue

        if tree_dirty:
            tree = cKDTree(np.array(accepted_xy))
            tree_dirty = False

        # Query: find all accepted picks within max possible exclusion radius
        # We need to check against each accepted pick's radius, so we query
        # a generous radius and then check precisely.
        max_r = max(accepted_r)
        query_radius = beta * max_r + r_cand   # conservative upper bound

        nearby_indices = tree.query_ball_point([x, y], r=query_radius)

        suppressed = False
        for j in nearby_indices:
            dist = float(np.hypot(x - accepted_xy[j][0], y - accepted_xy[j][1]))
            if dist < beta * accepted_r[j]:
                suppressed = True
                break

        if not suppressed:
            _accept(accepted_xy, accepted_r, accepted_idx, x, y, r_cand, i)
            tree_dirty = True

    if not accepted_idx:
        return np.empty((0, 5), dtype=np.float32)

    return cands[accepted_idx]


def _accept(
    accepted_xy: List,
    accepted_r: List,
    accepted_idx: List,
    x: float,
    y: float,
    r: float,
    idx: int,
) -> None:
    accepted_xy.append([x, y])
    accepted_r.append(r)
    accepted_idx.append(idx)
