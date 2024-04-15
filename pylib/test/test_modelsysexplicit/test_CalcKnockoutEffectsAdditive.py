import numpy as np
import pytest

from pylib.modelsys_explicit import CalcKnockoutEffectsAdditive


def test_CalcKnockoutEffectsAdditive_initialization():
    additive_array = np.array([0.1, -0.2, 0.0, 0.3])
    calc = CalcKnockoutEffectsAdditive(additive_array)
    assert np.array_equal(calc._additive_array, additive_array)


def test_CalcKnockoutEffectsAdditive_call():
    additive_array = np.array([0.1, -0.2, 0.0, 0.3])
    knockout = np.array([1, 0, 1, 1])
    expected_result = 0.1 + 0.3
    calc = CalcKnockoutEffectsAdditive(additive_array)
    assert np.isclose(calc(knockout), expected_result)


@pytest.mark.parametrize(
    "knockout, expected",
    [
        (np.array([1, 1, 1, 1]), 0.1 - 0.2 + 0.3),  # all elements knocked out
        (np.array([0, 0, 0, 0]), 0.0),  # no elements knocked out
    ],
)
def test_CalcKnockoutEffectsAdditive_edge_cases(
    knockout: np.array, expected: float
):
    additive_array = np.array([0.1, -0.2, 0.0, 0.3])
    calc = CalcKnockoutEffectsAdditive(additive_array)
    assert calc(knockout) == expected


def test_CalcKnockoutEffectsAdditive_empty():
    additive_array = np.array([])
    knockout = np.array([])
    calc = CalcKnockoutEffectsAdditive(additive_array)
    assert calc(knockout) == 0
