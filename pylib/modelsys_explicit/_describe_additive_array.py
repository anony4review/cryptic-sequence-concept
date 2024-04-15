import numpy as np
import pandas as pd


def describe_additive_array(additive_array: np.array) -> pd.DataFrame:
    """Create a dataframe describing additive effects of each genome site."""
    return pd.DataFrame(
        {
            "site": np.arange(len(additive_array)),
            "additive site": additive_array.astype(bool),
            "additive effect": additive_array,
        },
    )
