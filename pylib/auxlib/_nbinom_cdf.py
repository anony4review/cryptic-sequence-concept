from scipy import stats as scipy_stats


def nbinom_cdf(
    num_trials: int,
    num_successes: int,
    success_probability: float,
) -> float:
    """Compute the cumulative distribution function (CDF) of the negative
    binomial distribution.

    This function calculates the probability of having a number of failures
    before achieving a specified number of successes in a sequence of
    independent and identically distributed Bernoulli trials.

    Provides interface to scipy nbinom, but with a more convenient
    parameterization convention.

    Parameters
    ----------
    num_trials : int
        Total number of trials (successes + failures).
    num_successes : int
        The number of successes to be achieved.
    success_probability : float
        Probability of success in each trial.

    Returns
    -------
    float
        The cumulative probability of achieving the specified number of
        successes in the given number of trials.
    """
    # note: nbinom handles negative num failures okay (i.e., return zero)
    return scipy_stats.nbinom.cdf(
        num_trials - num_successes,  # num failures
        num_successes,  # num successes
        success_probability,  # success probability
    )
