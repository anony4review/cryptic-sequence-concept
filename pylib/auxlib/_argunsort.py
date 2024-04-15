import numpy as np


# adapted from https://stackoverflow.com/a/41558148/17332200
def argunsort(s: np.array) -> np.array:
    """Calculates the permutation necessary to undo a sort given the argsort
    array `s`."""
    n = s.size
    if n:
        u = np.empty(n, dtype=np.int64)
        u[s] = np.arange(n)
        return u
    else:
        return np.array([], dtype=int)
