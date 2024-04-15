import numpy as np
import pytest

from pylib.modelsys_explicit import create_additive_array


@pytest.mark.parametrize("effect_prevalence", [1.0, 0.5, 0.24, 0.0])
def test_create_additive_array_prevalence_rate(effect_prevalence: float):
    num_sites = 10
    array = create_additive_array(num_sites, effect_prevalence)
    assert len(array) == num_sites
    assert np.count_nonzero(array) == int(num_sites * effect_prevalence)

    if effect_prevalence:
        while np.array_equal(
            create_additive_array(num_sites, effect_prevalence), array
        ):
            pass


@pytest.mark.parametrize("effect_prevalence", [0, 1, 3, 10])
def test_create_additive_array_prevalence_count(effect_prevalence: int):
    num_sites = 10
    array = create_additive_array(num_sites, effect_prevalence)
    assert len(array) == num_sites
    assert np.count_nonzero(array) == effect_prevalence

    if effect_prevalence:
        while np.array_equal(
            create_additive_array(num_sites, effect_prevalence), array
        ):
            pass


def test_edge_cases():
    assert len(create_additive_array(0, 0.5)) == 0
    assert len(create_additive_array(1, 0.5)) == 1
    assert len(create_additive_array(1, 1)) == 1
