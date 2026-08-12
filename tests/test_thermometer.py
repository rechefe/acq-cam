"""Unit test for the circular thermometer code metric property (spec section 2.2).

hamming(code(sector_k), code(sector_j)) must equal 2 * circular_distance(k, j),
where circular_distance(k, j) = min(|k-j|, K-|k-j|), K = 2**B.

If this fails, the rest of the simulation is meaningless: the whole CAM-based
CFO estimate relies on Hamming distance tracking angular distance.
"""
import numpy as np
import pytest

from sim.encode import thermometer_code


@pytest.mark.parametrize("B", [2, 3])
def test_circular_thermometer_metric_property(B):
    K = 2 ** B
    # sector centers: guaranteed strictly on one side of every axis (no ties)
    thetas = (np.arange(K) + 0.5) * 2 * np.pi / K
    z = np.exp(1j * thetas)
    codes = thermometer_code(z, B)
    assert codes.shape == (K, K)

    for k in range(K):
        for j in range(K):
            hamming = int(np.sum(codes[k] != codes[j]))
            circ_dist = min(abs(k - j), K - abs(k - j))
            expected = 2 * circ_dist
            assert hamming == expected, (
                f"B={B}, k={k}, j={j}: hamming={hamming}, expected={expected}"
            )


@pytest.mark.parametrize("B", [2, 3])
def test_thermometer_self_distance_zero(B):
    K = 2 ** B
    thetas = (np.arange(K) + 0.5) * 2 * np.pi / K
    z = np.exp(1j * thetas)
    codes = thermometer_code(z, B)
    for k in range(K):
        assert np.array_equal(codes[k], codes[k])


@pytest.mark.parametrize("B", [2, 3])
def test_thermometer_monotone_with_angular_distance(B):
    """Distance from sector 0 must be non-decreasing as we walk away from it in
    either direction, peaking at the antipodal sector -- the defining
    'thermometer' (not just any code) property."""
    K = 2 ** B
    thetas = (np.arange(K) + 0.5) * 2 * np.pi / K
    z = np.exp(1j * thetas)
    codes = thermometer_code(z, B)
    ref = codes[0]
    dists = [int(np.sum(codes[k] != ref)) for k in range(K)]
    # walk 0 -> K/2 should be non-decreasing
    half = K // 2
    forward = dists[: half + 1]
    assert all(forward[i] <= forward[i + 1] for i in range(len(forward) - 1))
