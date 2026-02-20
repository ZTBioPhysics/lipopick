"""
Unit tests for MPI helper functions.

These test the pure (non-MPI) helpers in lipopick.mpi and can run with
plain ``pytest`` — no MPI runtime required.
"""
import numpy as np
import pytest

from lipopick.mpi import _scatter_work, _merge_results, _safe_process_one


# ===================================================================== #
# _scatter_work
# ===================================================================== #

class TestScatterWork:
    def test_even_split(self):
        chunks = _scatter_work([1, 2, 3, 4, 5, 6], 3)
        assert len(chunks) == 3
        assert chunks[0] == [1, 4]
        assert chunks[1] == [2, 5]
        assert chunks[2] == [3, 6]

    def test_uneven_split(self):
        chunks = _scatter_work([1, 2, 3, 4, 5], 3)
        assert len(chunks) == 3
        assert chunks[0] == [1, 4]
        assert chunks[1] == [2, 5]
        assert chunks[2] == [3]

    def test_more_ranks_than_items(self):
        chunks = _scatter_work([1, 2], 5)
        assert len(chunks) == 5
        assert chunks[0] == [1]
        assert chunks[1] == [2]
        assert chunks[2] == []
        assert chunks[3] == []
        assert chunks[4] == []

    def test_single_rank(self):
        items = [10, 20, 30]
        chunks = _scatter_work(items, 1)
        assert len(chunks) == 1
        assert chunks[0] == items

    def test_empty_input(self):
        chunks = _scatter_work([], 4)
        assert len(chunks) == 4
        assert all(c == [] for c in chunks)

    def test_preserves_all_items(self):
        items = list(range(17))
        chunks = _scatter_work(items, 5)
        flat = [x for c in chunks for x in c]
        assert sorted(flat) == items


# ===================================================================== #
# _merge_results
# ===================================================================== #

class TestMergeResults:
    def test_flatten_and_sort(self):
        gathered = [
            [{"path": "c.mrc", "n_picks": 3}],
            [{"path": "a.mrc", "n_picks": 1}, {"path": "b.mrc", "n_picks": 2}],
        ]
        merged = _merge_results(gathered)
        assert len(merged) == 3
        assert [r["path"] for r in merged] == ["a.mrc", "b.mrc", "c.mrc"]

    def test_handle_empty_ranks(self):
        gathered = [
            [{"path": "b.mrc", "n_picks": 2}],
            [],
            [{"path": "a.mrc", "n_picks": 1}],
            [],
        ]
        merged = _merge_results(gathered)
        assert len(merged) == 2
        assert merged[0]["path"] == "a.mrc"

    def test_all_empty(self):
        gathered = [[], [], []]
        merged = _merge_results(gathered)
        assert merged == []

    def test_single_rank(self):
        gathered = [[{"path": "x.mrc", "n_picks": 5}]]
        merged = _merge_results(gathered)
        assert len(merged) == 1


# ===================================================================== #
# _safe_process_one — error handling
# ===================================================================== #

class TestSafeProcessOne:
    def test_returns_error_dict_on_failure(self, tmp_path):
        """Passing a nonexistent path should return an error dict, not raise."""
        result = _safe_process_one(
            mic_path=str(tmp_path / "nonexistent.mrc"),
            outdir=str(tmp_path),
            cfg=None,  # will be set to default inside process_micrograph
            rank=0, size=1, idx=1, total=1,
            verbose=False,
        )
        assert result["n_picks"] == 0
        assert "error" in result
        assert isinstance(result["picks"], np.ndarray)
        assert result["picks"].shape == (0, 5)
