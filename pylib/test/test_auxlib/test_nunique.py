import numpy as np
import pytest

from pylib.auxlib._nunique import nunique


@pytest.mark.parametrize(
    "input_array, expected_result",
    [
        (np.array([1, 2, 3, 4, 5]), 5),  # Test with an array of unique values
        (
            np.array([1, 1, 2, 2, 3, 3]),
            3,
        ),  # Test with an array of repeated values
        (np.array([]), 0),  # Test with an empty array
        (np.array([1]), 1),  # Test with an array containing a single element
        (
            np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4]),
            4,
        ),  # Test with more repeated values
        (
            np.array([1, 2, 3, 4, 5, 5, 4, 3, 2, 1]),
            5,
        ),  # Test with reversed order
    ],
)
def test_nunique(input_array, expected_result):
    result = nunique(input_array)
    assert result == expected_result


def test_nunique_with_strings():
    input_array = np.array(["apple", "banana", "apple", "cherry", "banana"])
    expected_result = 3  # Counting distinct strings
    result = nunique(input_array)
    assert result == expected_result
