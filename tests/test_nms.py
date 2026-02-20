"""Tests for nms.py — size-aware non-maximum suppression."""
import numpy as np
import pytest

from lipopick.nms import size_aware_nms


def make_candidates(rows):
    """
    rows: list of (x, y, sigma, score, ds)
    """
    return np.array(rows, dtype=np.float32)


def test_nms_empty():
    cands = make_candidates([])
    if len(cands) == 0:
        cands = np.empty((0, 5), dtype=np.float32)
    result = size_aware_nms(cands)
    assert result.shape[0] == 0


def test_nms_single_pick():
    cands = make_candidates([[100.0, 100.0, 50.0, 1.0, 1]])
    result = size_aware_nms(cands)
    assert result.shape[0] == 1


def test_nms_two_far_picks_both_survive():
    """Two picks far apart (beyond any exclusion zone) should both survive."""
    # sigma=50 → radius≈70px, exclusion at beta=0.8 is 56px
    # Distance = 300px >> 56px → both survive
    cands = make_candidates([
        [100.0, 100.0, 50.0, 2.0, 1],
        [400.0, 100.0, 50.0, 1.0, 1],
    ])
    result = size_aware_nms(cands, beta=0.8)
    assert result.shape[0] == 2


def test_nms_duplicate_suppressed():
    """Two picks at nearly the same location: only the higher-score one survives."""
    cands = make_candidates([
        [100.0, 100.0, 50.0, 2.0, 1],   # higher score → accepted first
        [102.0, 101.0, 50.0, 1.0, 1],   # very close → suppressed
    ])
    result = size_aware_nms(cands, beta=0.8)
    assert result.shape[0] == 1
    assert result[0, 3] == pytest.approx(2.0)   # the higher-score one


def test_nms_small_inside_large_suppressed():
    """
    A small candidate located inside a large accepted particle should be suppressed.
    Large particle: sigma=140 → r≈198px, exclusion at beta=0.8 is 158px.
    Small candidate 80px away → inside exclusion zone → suppressed.
    """
    large_sigma = 140.0    # diameter ≈ 396px, radius ≈ 198px
    small_sigma = 30.0     # diameter ≈ 85px

    large_x, large_y = 300.0, 300.0
    small_x, small_y = 360.0, 300.0   # 60px away, well inside large exclusion

    cands = make_candidates([
        [large_x, large_y, large_sigma, 5.0, 1],    # large, high score → accepted first
        [small_x, small_y, small_sigma, 3.0, 1],    # small, inside large → suppressed
    ])
    result = size_aware_nms(cands, beta=0.8)
    assert result.shape[0] == 1
    assert result[0, 2] == pytest.approx(large_sigma)


def test_nms_two_medium_separated_both_survive():
    """Two medium particles just outside each other's exclusion zone both survive."""
    sigma = 70.0   # radius ≈ 99px, exclusion at beta=0.8 ≈ 79px
    # Place them 200px apart — outside exclusion zone
    cands = make_candidates([
        [200.0, 200.0, sigma, 2.0, 1],
        [400.0, 200.0, sigma, 1.5, 1],
    ])
    result = size_aware_nms(cands, beta=0.8)
    assert result.shape[0] == 2


def test_nms_edge_mask():
    """Candidates marked by edge_mask should be excluded."""
    cands = make_candidates([
        [10.0, 10.0, 50.0, 5.0, 1],     # near edge
        [300.0, 300.0, 50.0, 2.0, 1],   # interior
    ])
    edge_mask = np.array([True, False])
    result = size_aware_nms(cands, beta=0.8, edge_mask=edge_mask)
    assert result.shape[0] == 1
    assert result[0, 0] == pytest.approx(300.0)
