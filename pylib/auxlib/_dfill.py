import numpy as np


# adapted from https://stackoverflow.com/a/41558148/17332200
def dfill(a: np.array) -> np.array:
    """Repeat the indices of the start of each unique sequence in a 1D numpy
    array.

    This function takes an array and returns the positions where the array
    changes and repeats that index position until the next change.

    Parameters
    ----------
    a : np.array
        A 1D numpy array.

    Returns
    -------
    np.array
        A 1D numpy array of integers.

        Each element is the starting index of a unique sequence in the input
        array 'a', repeated to match the length of that sequence.

    Examples
    --------
    >>> import numpy as np
    >>> a = np.array(['a', 'a', 'a', 'b', 'a', 'a', 'a', 'c', 'a', 'a', 'a',
        'd', 'a', 'a', 'a', 'c'])
    >>> dfill(a)
    array([0, 0, 0, 3, 4, 4, 4, 7, 8, 8, 8, 11, 12, 12, 12, 15])
    >>> a = np.array(['a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a',
    'a', 'b', 'c', 'c', d'])
    >>> dfill(a)
    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 13, 13, 15])

    Notes
    -----
    - The function works only with 1D numpy arrays.
    - The function does not handle NaNs or missing values.
    - The input array should contain comparable elements; otherwise, the
       behavior is undefined.
    """
    n = a.size
    if n:
        b = np.concatenate([[0], np.where(a[:-1] != a[1:])[0] + 1, [n]])
        return np.arange(n)[b[:-1]].repeat(np.diff(b))
    else:
        return np.array([], dtype=int)
