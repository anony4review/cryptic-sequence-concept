import typing

from UniformSumDistribution import UniformSumDistribution
import numpy as np


class IrwinHallDistribution:
    """Implementation of Irwin-Hall distribution (sum of `n` iid uniform random
    variables).

    Provides capability to parametrize bounds of underlying uniform distributions, i.e., to replace unit 0 to 1 bounds.
    """

    _ptp: float  # peak-to-peak (maximum - minimum) of a & b
    _lb: float  # smaller of a & b
    _distn: UniformSumDistribution  # underlying implementation

    def __init__(
        self: "IrwinHallDistribution",
        n: int,
        a: float,
        b: float,
    ) -> None:
        """Initialize.

        Parameters
        ----------
        n : int
            The number of uniform random variables to sum.
        a : float
            The lower bound of underlying uniform distribution.
        b : float
            The upper bound of underlying uniform distribution.
        """
        self._ptp = np.ptp([a, b])
        self._lb = min(a, b) * n
        self._distn = UniformSumDistribution(n)

    def cdf(
        self: "IrwinHallDistribution",
        x: typing.Union[float, np.array],
        *args,
        **kwargs
    ) -> float:
        """Compute the cumulative distribution function of the Irwin-Hall
        distribution.

        Parameters
        ----------
        x : float or np.ndarray
            The value(s) at which to evaluate the CDF.

        Returns
        -------
        float or np.ndarray
            The CDF evaluated at `x`.
        """
        x = (x - self._lb) / self._ptp
        res = self._distn.cdf(x, *args, **kwargs)
        try:  # correct for numerical instability at extreme values
            return np.maximum.accumulate(np.clip(res, 0, 1))
        except TypeError:
            return np.clip(res, 0, 1)
