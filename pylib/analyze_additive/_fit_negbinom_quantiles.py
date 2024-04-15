import typing
import warnings

import numpy as np
from scipy.optimize import brute as scipy_brute  # grid search

from ..auxlib._nbinom_cdf import nbinom_cdf


def fit_negbinom_quantiles(
    distribution_values: typing.Sequence[int],
    cumulative_probabilities: typing.Sequence[float],
    r_upper_bound: int = 100,
    p_upper_bound: float = 1.0,
) -> dict:
    """Fit a negative binomial distribution to a set of cumulative distribution
    distribution_values.

    Parameters
    ----------
    distribution_values : typing.Sequence[float]
        A sequence of distribution values, i.e., counts.
    cumulative_probabilities : typing.Sequence[float]
        Cumulative probabilities corresponding to counts, i.e., quantiles..

    Returns
    -------
    dict
        The r and p parameters of the fitted negative binomial distribution, fit quantiles for `distribution_values`, and fit error.
    """
    if len(distribution_values) != len(cumulative_probabilities):
        raise ValueError(
            "distribution_values and cumulative_probabilities lengths differ.",
        )
    if len(distribution_values) <= 1:
        warnings.warn("Distribution fit is underspecified.")
    if (
        np.isnan(distribution_values).any()
        or np.isnan(cumulative_probabilities).any()
    ):
        raise ValueError("NaNs are not allowed.")
    if not np.array_equal(
        np.clip(cumulative_probabilities, 0, 1).tolist(),
        cumulative_probabilities,
    ):
        raise ValueError(
            "cumulative_probabilities must be in the range [0, 1]."
        )

    # Objective function to minimize
    pgranule = 100
    pnorm = pgranule / p_upper_bound

    def error_function(params) -> float:
        r, p = params
        assert r > 0 and p > 0
        error = sum(
            (nbinom_cdf(v, int(r), p / pnorm) - q) ** 2
            for v, q in zip(
                distribution_values, cumulative_probabilities, strict=True
            )
        )
        assert not np.isnan(error)
        return error

    # Range for r and p
    r_search = slice(1, r_upper_bound + 1, 1)
    p_search = slice(1, pgranule + 1, 1)

    # Using brute to find the optimal parameters --- minimize error
    result = scipy_brute(error_function, (r_search, p_search), finish=None)

    best_r, best_p = int(result[0]), result[1]
    assert best_r > 0
    assert 0 <= best_p / pnorm <= 1.0
    fit_quantiles = [
        nbinom_cdf(q, best_r, best_p / pnorm) for q in distribution_values
    ]

    # warn if the best parameters are at the edge of the search range
    if best_p >= p_search.stop - p_search.step:
        warnings.warn(
            "p is at the upper bound of the search range. "
            "Consider increasing the upper bound."
        )
    elif best_p <= p_search.start:
        warnings.warn(
            "p is at the lower bound of the search range. "
            "Consider decreasing the upper bound."
        )
    if best_r >= r_search.stop - r_search.step:
        warnings.warn(
            "r is at the upper bound of the search range. "
            "Consider increasing the upper bound."
        )
    elif best_r <= r_search.start:
        warnings.warn(
            "r is at the lower bound of the search range. "
            "Consider decreasing the upper bound."
        )

    return {
        "r": best_r,
        "p": best_p / pnorm,
        "fit quantiles": fit_quantiles,
        "error": error_function([best_r, best_p]),
    }
