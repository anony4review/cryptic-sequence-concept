import numpy as np

from pylib.analyze_additive import fit_negbinom_quantiles
from pylib.auxlib._nbinom_cdf import nbinom_cdf


def test_fit_negbinom_quantiles_exact():
    n, p = 10, 0.44
    counts = np.array([20, 30, 35, 40])
    cumulative_probabilities = nbinom_cdf(counts, n, p)
    result = fit_negbinom_quantiles(counts, cumulative_probabilities)
    assert isinstance(result, dict)
    assert "r" in result and "p" in result
    assert 0.35 < result["p"] < 0.45
    assert 8 < result["r"] < 12, result["fit quantiles"]
    assert len(result["fit quantiles"]) == 4


def test_fit_negbinom_quantiles_noised():
    n, p = 3, 0.05
    counts = np.array([30, 40, 50])
    cumulative_probabilities = nbinom_cdf(counts, n, p)
    noise = np.array([0.01, -0.02, 0.0])
    result = fit_negbinom_quantiles(
        counts, np.clip(cumulative_probabilities + noise, 0, 1)
    )
    assert isinstance(result, dict)
    assert "r" in result and "p" in result
    assert 2 < result["r"] < 5
    assert 0.01 < result["p"] < 0.1
    assert len(result["fit quantiles"]) == 3
