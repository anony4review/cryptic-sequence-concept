import numpy as np
import pytest

from pylib.analyze_additive import sample_knockout


@pytest.mark.parametrize(
    "num_sites",
    [10, 20],
)
@pytest.mark.parametrize(
    "num_knockouts",
    [0, 2, 9],
)
def test_sample_knockout(num_sites: int, num_knockouts: int):
    result = sample_knockout(num_knockouts, num_sites)
    assert result.shape == (num_sites,)
    assert result.sum() == num_knockouts
    if num_knockouts:
        while np.array_equal(
            sample_knockout(num_knockouts, num_sites), result
        ):
            pass
