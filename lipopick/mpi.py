"""
MPI-parallel batch processing for lipopick.

Distributes micrographs across MPI ranks using scatter/gather.
Each rank processes its subset independently (embarrassingly parallel),
then rank 0 merges results, computes QC flags, and writes summary outputs.

Requires ``mpi4py``:  ``pip install lipopick[mpi]``

Usage::

    mpirun -np 32 lipopick-mpi -i /path/to/micrographs/ -o /path/to/outputs/ \\
        --dmin 50 --dmax 100 --pixel-size 3.0341 --threshold-percentile 80 \\
        --nms-beta 1.3 --refine
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np

from .config import PickerConfig
from .io import list_micrographs, write_combined_csv
from .pipeline import (
    process_micrograph,
    _compute_qc_flags,
    _write_qc_csv,
    _generate_batch_summary,
    _picks_array_to_dicts,
    plt_close,
)


# ------------------------------------------------------------------ #
# Pure helpers (no MPI dependency — easily testable)
# ------------------------------------------------------------------ #

def _scatter_work(items: list, size: int) -> List[list]:
    """
    Round-robin split of *items* into *size* chunks.

    Returns a list of *size* lists.  Handles ``len(items) < size``
    (some chunks will be empty) and empty input.

    >>> _scatter_work([1,2,3,4,5], 3)
    [[1, 4], [2, 5], [3]]
    """
    chunks: List[list] = [[] for _ in range(size)]
    for i, item in enumerate(items):
        chunks[i % size].append(item)
    return chunks


def _merge_results(gathered: List[list]) -> List[dict]:
    """
    Flatten list-of-lists from ``comm.gather()`` and sort by path.

    *gathered* is a list where each element is a list of result dicts
    from one rank.  Returns a single sorted list.
    """
    flat = []
    for rank_results in gathered:
        if rank_results:
            flat.extend(rank_results)
    flat.sort(key=lambda r: r["path"])
    return flat


def _safe_process_one(
    mic_path: str,
    outdir: str,
    cfg: PickerConfig,
    rank: int,
    size: int,
    idx: int,
    total: int,
    verbose: bool,
) -> dict:
    """
    Wrapper around ``process_micrograph()`` with error handling.

    On failure, returns a minimal error dict so one bad micrograph
    does not crash the entire MPI job.
    """
    stem = Path(mic_path).stem
    if verbose:
        print(f"[rank {rank}/{size}] ({idx}/{total}) {stem}", flush=True)

    try:
        result = process_micrograph(mic_path, outdir, cfg=cfg, verbose=False)
        return result
    except Exception as e:
        tb = traceback.format_exc()
        if verbose:
            print(f"[rank {rank}/{size}] ERROR on {stem}: {e}\n{tb}", flush=True)
        return {
            "path": str(mic_path),
            "n_picks": 0,
            "time_s": 0.0,
            "picks_csv": None,
            "picks": np.empty((0, 5), dtype=np.float32),
            "image_std": 0.0,
            "dynamic_range": 0.0,
            "image_shape": (0, 0),
            "error": str(e),
        }


# ------------------------------------------------------------------ #
# Main MPI entry point
# ------------------------------------------------------------------ #

def mpi_process_batch(
    mic_dir,
    outdir: str | Path,
    cfg: Optional[PickerConfig] = None,
    verbose: bool = True,
    show_mic: Optional[str] = None,
) -> List[dict]:
    """
    Process micrographs using MPI parallelism.

    Parameters
    ----------
    mic_dir : str, Path, or list
        Directory containing micrographs, or a pre-built list of paths
        (e.g. gathered from multiple input directories).

    Communication pattern::

        Rank 0: list paths -> _scatter_work() -> comm.scatter()
        All ranks: loop _safe_process_one() on local subset
        All ranks: comm.gather(local_results, root=0)
        Rank 0: _merge_results() -> QC flags -> write CSVs -> batch summary

    Returns the merged results list on rank 0, empty list on other ranks.
    """
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if cfg is None:
        cfg = PickerConfig()

    outdir = Path(outdir)

    # ── Rank 0: discover micrographs and distribute work ──────────
    if rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        if isinstance(mic_dir, (list, tuple)):
            mic_paths = [str(p) for p in sorted(mic_dir)]
        else:
            mic_paths = [str(p) for p in list_micrographs(mic_dir)]
        n_total = len(mic_paths)
        if verbose:
            print(f"[rank 0/{size}] Found {n_total} micrograph(s)",
                  flush=True)
        chunks = _scatter_work(mic_paths, size)
    else:
        chunks = None
        n_total = 0

    # Broadcast total count so every rank can print progress
    n_total = comm.bcast(n_total, root=0)

    # Scatter work: each rank gets its chunk
    local_paths = comm.scatter(chunks, root=0)

    # ── Each rank: process its subset ─────────────────────────────
    t0 = time.perf_counter()
    local_results = []
    for i, mic_path in enumerate(local_paths, 1):
        result = _safe_process_one(
            mic_path, str(outdir), cfg,
            rank=rank, size=size, idx=i, total=len(local_paths),
            verbose=verbose,
        )
        local_results.append(result)

    local_time = time.perf_counter() - t0
    if verbose:
        local_picks = sum(r["n_picks"] for r in local_results)
        print(f"[rank {rank}/{size}] Done: {local_picks} picks from "
              f"{len(local_results)} micrographs in {local_time:.1f}s", flush=True)

    # ── Gather all results on rank 0 ──────────────────────────────
    gathered = comm.gather(local_results, root=0)

    if rank != 0:
        return []

    # ── Rank 0: merge and write outputs ───────────────────────────
    wall_time = time.perf_counter() - t0
    results = _merge_results(gathered)

    total_picks = sum(r["n_picks"] for r in results)
    total_cpu = sum(r["time_s"] for r in results)
    n_errors = sum(1 for r in results if "error" in r)

    if verbose:
        print(f"\n[rank 0/{size}] Merged: {total_picks} picks across "
              f"{len(results)} micrographs ({wall_time:.1f}s wall, "
              f"{total_cpu:.1f}s CPU)", flush=True)
        if n_errors > 0:
            print(f"[rank 0/{size}] WARNING: {n_errors} micrograph(s) failed",
                  flush=True)

    # QC flagging
    _compute_qc_flags(results)
    n_pass = sum(1 for r in results if r["qc_pass"])
    n_flag = len(results) - n_pass
    if verbose:
        print(f"[rank 0/{size}] QC: {n_pass} PASS, {n_flag} FLAGGED", flush=True)

    # Combined CSV
    combined_rows: List[dict] = []
    for r in results:
        stem = Path(r["path"]).stem
        for row in _picks_array_to_dicts(r["picks"]):
            row["micrograph"] = stem
            combined_rows.append(row)
    write_combined_csv(combined_rows, outdir / "all_picks.csv")
    if verbose:
        print(f"[rank 0/{size}] Combined CSV: {outdir / 'all_picks.csv'}"
              f"  ({len(combined_rows)} rows)", flush=True)

    # QC CSV
    _write_qc_csv(results, outdir / "micrograph_qc.csv")
    if verbose:
        print(f"[rank 0/{size}] QC CSV: {outdir / 'micrograph_qc.csv'}", flush=True)

    # Batch summary figure
    _generate_batch_summary(results, outdir, cfg, wall_time, verbose=False,
                            show_mic=show_mic)
    plt_close()
    if verbose:
        print(f"[rank 0/{size}] Summary figure: {outdir / 'batch_summary.png'}",
              flush=True)

    return results
