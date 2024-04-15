import numpy as np

from pylib.modelsys_explicit import CalcKnockoutEffectsEpistasis


def test_initialization():
    matrix = np.array([[0, 1], [1, 0]])
    effect_thresh = 2
    effect_size = 2.0

    instance = CalcKnockoutEffectsEpistasis(matrix, effect_thresh, effect_size)
    assert np.array_equal(instance._epistasis_matrix, matrix)
    assert instance._effect_thresh == effect_thresh
    assert instance._effect_size == effect_size


def test_threshold_calculation():
    matrix = np.array([[2, 1, 2], [0, 2, 1]])
    instance = CalcKnockoutEffectsEpistasis(matrix, None)
    assert instance._effect_thresh == 3


def test_effect_size():
    matrix = np.array([[1, 0], [0, 1]])
    instance = CalcKnockoutEffectsEpistasis(matrix, 1, 1.5)
    assert instance._effect_size == 1.5


def test_effect_size_tuple():
    matrix = np.array([[1, 0, 1], [0, 1, 0]])
    instance = CalcKnockoutEffectsEpistasis(matrix, 1, effect_size=(1.0, 1.5))
    assert instance._effect_size.shape == (1,)


def test_knockout_effect_calculation():
    matrix = np.array([[1, 1, 0, 2], [2, 0, 0, 0]])
    instance = CalcKnockoutEffectsEpistasis(matrix, 2)
    assert instance(np.array([1, 1, 0, 0])) == 1.0
    assert instance(np.array([1, 1, 0, 1])) == 2.0
    assert instance(np.array([0, 0, 0, 0])) == 0.0
    assert instance(np.array([0, 0, 1, 0])) == 0.0
    assert instance(np.array([0, 0, 0, 1])) == 0.0
    assert instance(np.array([0, 0, 1, 1])) == 0.0


def test_knockout_effect_tuple():
    matrix = np.array([[1, 1, 0, 2], [2, 0, 0, 0]])
    instance = CalcKnockoutEffectsEpistasis(matrix, 2)
    instance._effect_size = np.array([3.0, 1.0])
    assert instance(np.array([1, 1, 0, 0])) == 3.0
    assert instance(np.array([1, 1, 0, 1])) == 4.0
    assert instance(np.array([0, 1, 0, 1])) == 0.0
    assert instance(np.array([0, 0, 0, 0])) == 0.0
    assert instance(np.array([0, 0, 1, 0])) == 0.0
    assert instance(np.array([0, 0, 0, 1])) == 0.0
    assert instance(np.array([0, 0, 1, 1])) == 0.0
    assert instance(np.array([1, 0, 1, 1])) == 1.0
    assert instance(np.array([1, 1, 1, 1])) == 4.0


def test_zero_epistasis_matrix():
    matrix = np.zeros((2, 2))
    instance = CalcKnockoutEffectsEpistasis(matrix, 1)
    knockout = np.array([1, 1])
    assert instance(knockout) == 0.0
