import numpy as np

from pylib.auxlib._dfill import dfill


def test_dfill1():
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
    expected = np.array([0, 0, 0, 3, 4, 4, 4, 7, 8, 8, 8, 11, 12, 12, 12, 15])
    result = dfill(a)
    assert np.array_equal(result, expected)


def test_dfill2():
    np.array([0, 0, 0, 3, 4, 4, 4, 7, 8, 8, 8, 11, 12, 12, 12, 15])
    a = np.array(
        [
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "b",
            "c",
            "c",
            "d",
        ]
    )
    expected = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 13, 13, 15])
    result = dfill(a)
    assert np.array_equal(result, expected)


def test_dfill_empty():
    a = np.array([])
    expected = np.array([])
    result = dfill(a)
    assert np.array_equal(result, expected)


def test_dfill_single():
    a = np.array(["a"])
    expected = np.array([0])
    result = dfill(a)
    assert np.array_equal(result, expected)
