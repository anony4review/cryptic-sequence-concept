import numpy as np

from pylib.auxlib._IrwinHallDistribution import IrwinHallDistribution


def test_IrwinHallDistribution_cdf():
    for n in 1, 10:
        dist = IrwinHallDistribution(n, 0, 1)
        assert np.isclose(dist.cdf(0), 0)
        assert 0 < dist.cdf(0.5) < 1
        assert np.isclose(dist.cdf(1 * n), 1)

        dist = IrwinHallDistribution(n, 1, 3)
        assert np.isclose(dist.cdf(1 * n), 0)
        assert 0 < dist.cdf(2 * n) < 1
        assert np.isclose(dist.cdf(3 * n), 1)
