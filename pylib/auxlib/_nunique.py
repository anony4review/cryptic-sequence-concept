import numpy as np
import pandas as pd


def nunique(a: np.array) -> int:
    """Count number of distinct elements in array."""
    # adapted from https://stackoverflow.com/a/48473236/17332200
    series = pd.Series(a.flat)
    return series.nunique()
