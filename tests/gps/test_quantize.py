"""The key-mapping gates: the metric property and the free rotation."""
import numpy as np
import pytest

from sim.gps.quantize import (adc_2bit, key_width, levels_to_thermo_bits,
                              quantize_to_key, rotate_key, row_state_bits)

MAPPINGS = ["sign", "thermometer"]


def test_thermometer_hamming_is_l1_distance_in_levels():
    """The load-bearing metric property: Hamming distance between thermometer
    words equals the L1 distance between the ADC levels they encode."""
    bits = levels_to_thermo_bits(np.arange(4))
    for a in range(4):
        for b in range(4):
            assert (bits[a] != bits[b]).sum() == abs(a - b)


def test_adc_thresholds_place_levels_symmetrically():
    x = np.array([-5.0, -1.5, -0.5, 0.5, 1.5, 5.0])
    assert np.array_equal(adc_2bit(x, sigma=1.0), [0, 0, 1, 2, 3, 3])


@pytest.mark.parametrize("mapping", MAPPINGS)
@pytest.mark.parametrize("quadrant", range(4))
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_rotation_is_exact_bit_permutation(mapping, quadrant, seed):
    """Rotating the key by k*90 degrees must equal quantizing the rotated
    waveform, bit for bit. This is what makes the theta0 sweep free."""
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 1, 128) + 1j * rng.normal(0, 1, 128)
    direct = quantize_to_key(r * np.exp(1j * np.pi / 2 * quadrant), 1.0, mapping)
    assert np.array_equal(direct, rotate_key(quantize_to_key(r, 1.0, mapping), quadrant, mapping))


@pytest.mark.parametrize("mapping", MAPPINGS)
def test_four_rotations_return_to_identity(mapping):
    rng = np.random.default_rng(3)
    k = quantize_to_key(rng.normal(0, 1, 64) + 1j * rng.normal(0, 1, 64), 1.0, mapping)
    out = k.copy()
    for _ in range(4):
        out = rotate_key(out, 1, mapping)
    assert np.array_equal(out, k)


@pytest.mark.parametrize("mapping,per_rail", [("sign", 1), ("thermometer", 3)])
def test_key_width(mapping, per_rail):
    assert key_width(100, mapping) == 100 * 2 * per_rail


def test_thermometer_widens_match_line_not_memory():
    """Rows are full scale, so stored state is 1 bit/rail whatever the mapping."""
    assert row_state_bits(2046) == 2 * 2046
    assert key_width(2046, "thermometer") == 3 * row_state_bits(2046)
