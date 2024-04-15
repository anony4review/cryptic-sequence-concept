import numpy as np
import pandas as pd


def describe_epistasis_matrix(epistasis_matrix: np.array) -> pd.DataFrame:
    """Create a dataframe describing epistatic effects of each genome site."""
    view = np.atleast_2d(epistasis_matrix).T
    return pd.DataFrame(
        {
            "site": np.arange(len(view)),
            "epistasis site": np.any(view.astype(bool), axis=1),
            "num epistasis effects": np.sum(view.astype(bool), axis=1),
        },
    )
