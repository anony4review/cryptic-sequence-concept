import numpy as np

from pylib.auxlib._nonzero import nonzero


def test_nonzero():
    a = np.array([0, 0, 0, 3, 4, 4, 4, 7, 8, 8, 8, 11, 12, 12, 12, 15])
    expected = np.array([3, 4, 4, 4, 7, 8, 8, 8, 11, 12, 12, 12, 15])
    result = nonzero(a)
    assert np.array_equal(result, expected)


def test_nonzero_single():
    a = np.array([0])
    expected = np.array([])
    result = nonzero(a)
    assert np.array_equal(result, expected)


def test_nonzero_empty():
    a = np.array([])
    expected = np.array([])
    result = nonzero(a)
    assert np.array_equal(result, expected)
