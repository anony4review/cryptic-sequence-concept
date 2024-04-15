import numpy as np
import pandas as pd

from pylib.modelsys_explicit import describe_additive_array


def test_describe_additive_array():
    additive_array = np.array([0.1, 0, 2.0])
    result = describe_additive_array(additive_array)
    expected = pd.DataFrame(
        {
            "site": [0, 1, 2],
            "additive site": [True, False, True],
            "additive effect": [0.1, 0.0, 2.0],
        },
    )
    pd.testing.assert_frame_equal(result, expected)


def test_describe_additive_array_empty():
    additive_array = np.array([])
    result = describe_additive_array(additive_array)
    expected = pd.DataFrame(
        {
            "site": np.array([], dtype=int),
            "additive site": np.array([], dtype=bool),
            "additive effect": np.array([], dtype=float),
        },
    )
    pd.testing.assert_frame_equal(result, expected)
