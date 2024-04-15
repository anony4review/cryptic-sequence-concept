import numpy as np

from pylib.auxlib._discrete_geomspace import discrete_geomspace


def test_discrete_geomspace():
    assert np.array_equal(
        discrete_geomspace(1, 10, 50), np.array([*range(1, 11)])
    )
    assert np.array_equal(
        discrete_geomspace(1, 10, 11), np.array([*range(1, 11)])
    )
    assert np.array_equal(
        discrete_geomspace(1, 10, 5), np.array([1, 2, 3, 5, 10])
    )
