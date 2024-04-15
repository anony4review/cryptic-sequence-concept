import numpy as np


def discrete_geomspace(start: float, stop: float, *args, **kwargs) -> np.array:
    """Return `n` integers spaced evenly on a log scale (a geometric progression), with any duplicates shifted up.

    Will return fewer than `n` values if `stop` - `start` is less than `n`.
    """
    vals = np.geomspace(start, stop, *args, **kwargs).astype(int)
    res = np.maximum(vals, vals.min() + np.arange(len(vals)))
    return res[np.clip(res, start, stop) == res].copy()
