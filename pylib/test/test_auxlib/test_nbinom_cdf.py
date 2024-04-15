import numpy as np
from scipy import stats as scipy_stats

from pylib.auxlib._nbinom_cdf import nbinom_cdf


def test_nbinom_cdf():
    assert np.allclose(
        nbinom_cdf(10, 3, 0.5), scipy_stats.nbinom.cdf(7, 3, 0.5)
    )

    assert np.allclose(
        nbinom_cdf(15, 5, 0.3), scipy_stats.nbinom.cdf(10, 5, 0.3)
    )
