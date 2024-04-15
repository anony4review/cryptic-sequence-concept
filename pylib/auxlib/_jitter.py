import numpy as np


# adapted from https://stackoverflow.com/a/64554001/17332200
def jitter(values: np.array, amount: float = 0.05) -> np.array:
    """Add random jitter to each element in a numpy array.

    Useful to deconflict scatter plots.

    Parameters
    ----------
    values : np.array
        An array of numeric values to which gaussian jitter will be added.
    amount : float, default 0.05
        The scale of the jitter as a proportion of the peak-to-peak (max - min)
        range of the values.

    Returns
    -------
    np.array
        The array with jitter added to each element.
    """
    if len(values) == 0:  # handle empty case
        return values
    scale = (np.ptp(values) or 0.1) * amount
    return np.random.normal(values, scale)
