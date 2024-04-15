import numpy as np

from pylib.auxlib._jitter import jitter


def test_jitter():
    test_values = np.array([1, 2, 3, 4, 5])

    jittered_values = jitter(test_values, 0.1)

    assert isinstance(jittered_values, np.ndarray)

    assert jittered_values.shape == test_values.shape

    # Check that the jittered values are within a reasonable range
    max_expected_jitter = np.ptp(test_values) * 0.5
    assert np.all(np.abs(jittered_values - test_values) < max_expected_jitter)


# Optionally, you can add more tests to check edge cases, for example:
def test_jitter_with_empty_array():
    assert np.array_equal(jitter(np.array([])), np.array([]))


def test_jitter_with_zero_amount():
    test_values = np.array([1, 2, 3, 4, 5])
    assert np.array_equal(jitter(test_values, amount=0), test_values)
