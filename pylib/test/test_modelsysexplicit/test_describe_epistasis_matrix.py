import numpy as np
import pandas as pd

from pylib.modelsys_explicit import describe_epistasis_matrix


def test_describe_epistasis_matrix_flat():
    epistasis_matrix = np.array([[1, 1, 0]])
    result = describe_epistasis_matrix(epistasis_matrix)
    expected = pd.DataFrame(
        {
            "site": [0, 1, 2],
            "epistasis site": [True, True, False],
            "num epistasis effects": [1, 1, 0],
        },
    )
    pd.testing.assert_frame_equal(result, expected)


def test_describe_epistasis_matrix_overlapped():
    epistasis_matrix = np.array([[1, 1, 0, 0], [2, 0, 2, 0]])
    result = describe_epistasis_matrix(epistasis_matrix)
    expected = pd.DataFrame(
        {
            "site": [0, 1, 2, 3],
            "epistasis site": [True, True, True, False],
            "num epistasis effects": [2, 1, 1, 0],
        },
    )
    pd.testing.assert_frame_equal(result, expected)


def test_describe_epistasis_matrix_empty():
    epistasis_matrix = np.array([])
    result = describe_epistasis_matrix(epistasis_matrix)
    expected = pd.DataFrame(
        {
            "site": np.array([], dtype=int),
            "epistasis site": np.array([], dtype=bool),
            "num epistasis effects": np.array([], dtype=int),
        },
    )
    print(result)
    pd.testing.assert_frame_equal(result, expected)
