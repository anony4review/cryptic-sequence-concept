import numpy as np

from pylib.auxlib._cumcount import cumcount


def test_cumcount():
    a = np.array(
        [
            "a",
            "a",
            "a",
            "b",
            "a",
            "a",
            "a",
            "c",
            "a",
            "a",
            "a",
            "d",
            "a",
            "a",
            "a",
            "c",
        ]
    )
    expected = np.array([0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 9, 10, 11, 1])
    result = cumcount(a)
    assert np.array_equal(result, expected)


def test_cumcount_empty():
    a = np.array([])
    expected = np.array([])
    result = cumcount(a)
    assert np.array_equal(result, expected)


def test_cumcount_single():
    a = np.array(["a"])
    expected = np.array([0])
    result = cumcount(a)
    assert np.array_equal(result, expected)
