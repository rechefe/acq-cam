"""CAM primitive correctness against a brute-force reference, and API surface check:
cam.py must not expose any distance-returning method (enforced by API, not discipline).
"""
import inspect

import numpy as np
import pytest

from sim.cam import CAM


def brute_force_match(rows, key, tau, mask=None):
    rows = np.asarray(rows).astype(bool)
    key = np.asarray(key).astype(bool)
    match = np.zeros(rows.shape[0], dtype=bool)
    for k in range(rows.shape[0]):
        mism = rows[k] != key
        if mask is not None:
            mism = mism & (~mask[k].astype(bool))
        d = int(mism.sum())
        match[k] = d <= tau
    return match


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_cam_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    N, W = 32, 100
    rows = rng.integers(0, 2, size=(N, W))
    key = rng.integers(0, 2, size=W)
    cam = CAM(rows)
    for tau in [0, 1, 5, 20, 50, W]:
        expected = brute_force_match(rows, key, tau)
        got = cam.query(key, tau)
        assert np.array_equal(got, expected), f"tau={tau} mismatch"


@pytest.mark.parametrize("seed", [0, 1])
def test_cam_ternary_mask_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    N, W = 16, 60
    rows = rng.integers(0, 2, size=(N, W))
    mask = rng.random((N, W)) < 0.2
    key = rng.integers(0, 2, size=W)
    cam = CAM(rows, mask=mask)
    for tau in [0, 3, 10, 30]:
        expected = brute_force_match(rows, key, tau, mask=mask)
        got = cam.query(key, tau)
        assert np.array_equal(got, expected), f"tau={tau} mismatch"


def test_cam_exact_match_at_tau_zero():
    rows = np.array([[0, 0, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]])
    cam = CAM(rows)
    m = cam.query(np.array([0, 0, 1, 1]), tau=0)
    assert list(m) == [True, False, False]


def test_cam_query_shape_validation():
    rows = np.zeros((4, 10))
    cam = CAM(rows)
    with pytest.raises(ValueError):
        cam.query(np.zeros(5), tau=1)


def test_cam_api_exposes_no_distance_method():
    """Enforce with the API, not discipline: no public method of CAM may return
    anything but the boolean match vector from query()."""
    public_methods = [
        name for name, _ in inspect.getmembers(CAM, predicate=inspect.isfunction)
        if not name.startswith("_")
    ]
    assert public_methods == ["query"], f"CAM exposes unexpected public methods: {public_methods}"

    rng = np.random.default_rng(0)
    rows = rng.integers(0, 2, size=(8, 20))
    cam = CAM(rows)
    result = cam.query(rng.integers(0, 2, size=20), tau=5)
    assert result.dtype == np.bool_
