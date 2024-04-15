import numpy as np


def nonzero(a: np.array) -> np.array:
    """Isolate nonzero elements."""
    return a[a.nonzero()]
