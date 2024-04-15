import numpy as np
import pandas as pd

from pylib.analyze_epistasis import jackknife_skeleton


def mock_test_knockout(knockout_mask: np.array) -> float:
    # Sensitive to knockouts at even sites
    return np.any(knockout_mask[::2]) * 1.25


def test_jackknife_skeleton():
    mock_skeleton = np.array([0, 1, 0, 0], dtype=bool)
    # Expected DataFrame structure
    expected_df = pd.DataFrame(
        {
            "jackknife dose": [2, 2, 2],
            "raw jackknife result": [1.25, 1.25, 0.0],
            "jackknife result": [1.25, 1.25, 1.0],
            "site": [0, 2, 3],
        }
    )

    # Running the jackknife_skeleton function
    result_df = jackknife_skeleton(mock_skeleton, mock_test_knockout)

    # Asserting that the result is as expected
    pd.testing.assert_frame_equal(result_df, expected_df)
