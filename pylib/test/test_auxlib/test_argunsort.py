import numpy as np

from pylib.auxlib._argunsort import argunsort


def test_argunsort():
    arr = np.array([3, 1, 4, 1, 5, 9, 2])

    s = np.argsort(arr)
    u = argunsort(s)

    # Apply the permutations to the sorted array
    restored_arr = arr[s][u]

    assert np.array_equal(restored_arr, arr)


def test_argunsort_empty():
    arr = np.array([])
    assert np.array_equal(argunsort(arr), arr)


def test_argunsort_single():
    arr = np.array([0])
    assert np.array_equal(argunsort(arr), np.array([0]))
