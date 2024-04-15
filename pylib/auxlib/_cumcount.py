import numpy as np

from ._argunsort import argunsort
from ._dfill import dfill


# adapted from https://stackoverflow.com/a/41558148/17332200
def cumcount(a: np.array) -> np.array:
    """Calculate cumulative count by value occurrences in a 1D array.

    This function computes the cumulative count of each unique element in a 1D numpy
    array 'a'. The count for each element starts at 0 and increments for each
    occurrence of that element.

    Parameters
    ----------
    a : np.array
        A 1D numpy array containing the elements for which the cumulative count is
        to be calculated.

    Returns
    -------
    np.array
        A 1D numpy array of the same length as 'a', where each element is the
        cumulative count of the corresponding element in 'a'.

    Examples
    --------
    >>> a = np.array(['a', 'a', 'a', 'b', 'a', 'a', 'a', 'c', 'a', 'a', 'a', 'd', 'a', 'a', 'a', 'c'])
    >>> cumcount(a)
    array([0, 1, 2, 0, 3, 4, 5, 0, 6, 7, 8, 0, 9, 10, 11, 1])
    """
    n = a.size
    s = a.argsort(kind="mergesort")
    i = argunsort(s)
    b = a[s]
    return (np.arange(n) - dfill(b))[i]
