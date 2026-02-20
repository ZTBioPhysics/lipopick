"""Tests for pyramid.py"""
import numpy as np
import pytest

from lipopick.pyramid import build_pyramid_level, sigma_range_for_level


def test_ds1_no_change():
    """ds=1 should return a float32 copy with same shape."""
    img = np.ones((100, 100), dtype=np.float32)
    out = build_pyramid_level(img, ds=1)
    assert out.shape == (100, 100)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, img)


def test_ds2_half_size():
    """ds=2 should approximately halve the image dimensions."""
    img = np.random.default_rng(0).random((200, 200)).astype(np.float32)
    out = build_pyramid_level(img, ds=2)
    assert out.dtype == np.float32
    assert abs(out.shape[0] - 100) <= 1
    assert abs(out.shape[1] - 100) <= 1


def test_ds4_quarter_size():
    """ds=4 should approximately quarter the image dimensions."""
    img = np.random.default_rng(0).random((400, 400)).astype(np.float32)
    out = build_pyramid_level(img, ds=4)
    assert out.dtype == np.float32
    assert abs(out.shape[0] - 100) <= 1
    assert abs(out.shape[1] - 100) <= 1


def test_sigma_range_for_level_ordering():
    """sigma_min < sigma_max, n_steps >= min_steps + 1."""
    sigma_min, sigma_max, n_steps = sigma_range_for_level(
        d_lo=150, d_hi=275, ds=1, k=1.10, min_steps=3
    )
    assert sigma_min > 0
    assert sigma_max > sigma_min
    assert n_steps >= 4


def test_sigma_range_ds2():
    """At ds=2, level-pixel sigmas should be half the full-res sigmas."""
    s_min_1, s_max_1, _ = sigma_range_for_level(150, 275, ds=1, k=1.10)
    s_min_2, s_max_2, _ = sigma_range_for_level(150, 275, ds=2, k=1.10)
    np.testing.assert_allclose(s_min_2, s_min_1 / 2, rtol=1e-5)
    np.testing.assert_allclose(s_max_2, s_max_1 / 2, rtol=1e-5)
