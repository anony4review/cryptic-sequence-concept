import itertools as it
import math
import typing

import numpy as np
from scipy import stats as scipy_stats

# adapted from
# Burnham, Kenneth P., and W. Scott Overton.
# "Robust estimation of population size when capture probabilities vary among
# animals." Ecology 60.5 (1979): 927-936.
# https://doi.org/10.2307/1936861


# fmt: off
def _NJ0(f: typing.Sequence[int]) -> float:
    S = sum(f)
    return S


def _NJ1(f: typing.Sequence[int]) -> float:
    S = sum(f)
    t = len(f)
    fdict = dict(enumerate(f))
    return S +  fdict.get(0, 0) * (t - 1) / t


def _NJ2(f: typing.Sequence[int]) -> float:
    S = sum(f)
    t = len(f)
    fdict = dict(enumerate(f))
    return (
        S
        + fdict.get(0, 0) * (2 * t - 3) / t
        - fdict.get(1, 0) * (t - 2) ** 2 / (t * (t - 1))
    )


def _NJ3(f: typing.Sequence[int]) -> float:
    S = sum(f)
    t = len(f)
    fdict = dict(enumerate(f))
    return (
        S
        + fdict.get(0, 0) * (3 * t - 6) / t
        - fdict.get(1, 0) * (3 * t ** 2 - 15 * t + 19) / (t * (t - 1))
        + fdict.get(2, 0) * (t - 3) ** 3 / (t * (t - 1) * (t -2))
    )


def _NJ4(f: typing.Sequence[int]) -> float:
    S = sum(f)
    t = len(f)
    fdict = dict(enumerate(f))
    return (
        S
        + fdict.get(0, 0) * (4 * t - 10) / t
        -  fdict.get(1, 0) * (6 * t ** 2 - 36 * t + 55) / (t * (t - 1))
        +  fdict.get(2, 0) * (4 * t ** 3 - 42 * t ** 2 + 148 * t - 175)
            / (t * (t - 1) * (t - 2))
        -  fdict.get(3, 0) * (t - 4) ** 4 / (t * (t - 1) * (t - 2) * (t - 3))
    )


def _NJ5(f: typing.Sequence[int]) -> float:
    S = sum(f)
    t = len(f)
    fdict = dict(enumerate(f))
    return (
        S
        + fdict.get(0, 0) * (5 * t - 15) / t
        - fdict.get(1, 0) * (10 * t ** 2 - 70 * t + 125) / (t * (t -1))
        + fdict.get(2, 0) * (10 * t ** 3 - 120 * t ** 2 + 485 * t - 660)
            / (t * (t - 1) * (t - 2))
        - fdict.get(3, 0) * ((t - 4) ** 5 - (t - 5) ** 5)
            / (t * (t - 1) * (t - 2) * (t - 3))
        + fdict.get(4, 0) * (t - 5) ** 5
            / (t * (t - 1) * (t - 2) * (t - 3) * (t - 4))
    )


def _NJk(k: int) -> typing.Callable:
    return [_NJ0, _NJ1, _NJ2, _NJ3, _NJ4, _NJ5][k]


def _alphaik(i: int, k: int):
    if k == 0:
        return lambda t: 0
    else:
        k -= 1

    if i > k:
        return lambda t: 0
    elif i == k:
        return lambda t: (t - i - 1) ** (i + 1) / math.prod(
            (t - c) for c in range(i + 1)
        ) * (-1) ** i
    else:
        return lambda t: {
            (1, 0): (2 * t - 3),
            (2, 0): (3 * t - 6),
            (2, 1): -(3 * t ** 2 - 15 * t + 19),
            (3, 0): (4 * t - 10),
            (3, 1): -(6 * t ** 2 - 36 * t + 55),
            (3, 2): (4 * t ** 3 - 42 * t ** 2 + 148 * t - 175),
            (4, 0): (5 * t - 15),
            (4, 1): -(10 * t ** 2 - 70 * t + 125),
            (4, 2): (10 * t ** 3 - 120 * t ** 2 + 485 * t - 660),
            (4, 3): -((t - 4) ** 5 - (t - 5) ** 5)
        }[(k, i)] / math.prod(
            (t - c) for c in range(i + 1)
        )

def _aik(i: int, k: int) -> typing.Callable:
    return lambda t: _alphaik(i, k)(t) + 1

def _NJk_v2(k: int) -> typing.Callable:

    def _NJ(f: typing.Sequence[int]) -> float:
        t = len(f)
        fdict = dict(enumerate(f))
        return sum(
            _aik(i, k)(t) * fdict.get(i, 0)
            for i in range(t)
        )

    return _NJ

def _bi(i: int, k: int) -> typing.Callable:
    return lambda t: _aik(i, k + 1)(t) - _aik(i, k)(t)


def _varNJk_diff(k: int) -> typing.Callable:

    def _var(f: typing.Sequence[int]) -> int:
        S = sum(f)
        t = len(f)
        fdict = dict(enumerate(f))
        return S / (S - 1) * (
            sum(_bi(i, k)(t) ** 2 * fdict.get(i, 0) for i in range(t))
            - (_NJk(k + 1)(f) - _NJk(k)(f)) ** 2 / S
        )

    return _var


def _Tk(k: int) -> typing.Callable:

    def _T(f: typing.Sequence[int]) -> int:
        return (
            (_NJk(k + 1)(f) - _NJk(k)(f))
            / _varNJk_diff(k)(f) ** (1 / 2)
        )

    return _T

def _varNjk(k: int) -> typing.Callable:

    def varNj(f: typing.Sequence[int]) -> float:
        t = len(f)
        return sum(
            _aik(i, k)(t) ** 2 * f_
            for i, f_ in enumerate(f)
        ) - _NJk(k)(f)

    return varNj


def _seNjk(k: int) -> typing.Callable:

    def seNj(f: typing.Sequence[int]) -> float:
        return _varNjk(k)(f) ** (1 / 2)

    return seNj


def _95CIk(k: int) -> typing.Callable:

    if k == 0:
        return lambda k: (np.nan, np.nan)

    def _95CI(f: typing.Sequence[int]) -> typing.Tuple[float, float]:
        est = _NJk(k)(f)
        err = 1.96 * _seNjk(k)(f)
        return (est - err, est + err)

    return _95CI


def test_standard_normal(value, alpha=0.05) -> bool:
    p_value = 2 * scipy_stats.norm.sf(abs(value))
    return p_value <= alpha


def reject_h0(f: typing.Sequence[int], k: int) -> bool:
    test_statistic = _Tk(k)(f)
    return test_standard_normal(test_statistic)


# TODO
# - implement "improved selection procedure" at end of reference
def jackknife_mark_recapture_estimate(
    f: typing.Sequence[int],
) -> typing.Tuple[float, typing.Tuple[float, float]]:
    """Estimate the population size in mark-recapture studies with multiple
    recapture events using a nonparametric method based on the generalized
    jackknife approach that is robust to variability between population
    members' capture probabilities (e.g., trap-shyness).

    Parameters
    ----------
    f : typing.Sequence[int]
        A sequence of integers where the nth position denotes how many
        individuals were captured n times, up through the number of capture
        events (even if zero individuals were captured in all events).

        The sequence sum should correspond to the net unique individuals
        captured over the course of the study. The length of the sequence should
        correspond to the number of recapture events performed.

    Returns
    -------
    float, Tuple[float, float]
        An estimated value of the population size and 95% confidence interval.

        If estimate equals the number of distinct observed individuals, a valid
        confidence interval cannot be computed so returned confidence interval
        is `(nan, nan)`.

    Notes
    -----
    The estimation procedure is designed for closed population mark-recapture
    studies where individual capture probabilities are assumed to be constant
    over time but may vary among individuals.

    References
    ----------
    Amstrup, Steven C., Trent L. McDonald, and Bryan FJ Manly, eds. Handbook of
    capture-recapture analysis. Princeton University Press, 2010.

    Burnham, K. P., & Overton, W. S. (1979). Robust Estimation of Population
    Size When Capture Probabilities Vary Among Animals. Ecology, 60(5), 927-936.

    Examples
    --------
    >>> capture_frequencies = [
    ...     43, 16, 8, 6, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    >>> jackknife_mark_recapture_estimate(capture_frequencies)
    158.6
    """
    for k in it.count():
        if reject_h0(f, k):
            continue
        else:
            return _NJk(k)(f), _95CIk(k)(f)
