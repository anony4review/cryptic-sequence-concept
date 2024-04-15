import typing

import numpy as np
import pandas as pd


def jackknife_skeleton(
    skeleton: np.array,
    test_knockout: typing.Callable[[np.array], bool],
) -> pd.DataFrame:
    """Perform a jackknife analysis on a given knockout skeleton to assess the
    fitness result of removing each non-knocked out site.

    This function iteratively tests each site that was not knocked out in the
    original skeletonization process. It assesses whether knocking out these
    sites would lead to a detectable knockout effect, and the magnitude of the
    effect.

    Parameters
    ----------
    skeleton : np.array
        An array representing the knockout skeleton from, e.g.,
        `skeletonize_naive`.

        Each element corresponds to a genome site, where 0 indicates no knockout, and a positive integer indicates that the site was knocked out.

    test_knockout : typing.Callable[[np.array], bool]
        A function that tests the effect of a knockout. It takes a boolean mask
        as input (with True representing a knockout) and returns True if a
        knockout effect was detected, and False otherwise.

    Returns
    -------
    pd.DataFrame
        A DataFrame with rows corresponding to each retained site within the
        skeleton genome.

    Notes
    -----
    - Skeletonized genomes are knockout sets that cannot be extended further
      without a detectable fitness effect.
    - Because sensitivity is expected at each remaining site in a genome
      skeleton, the 'jackknife result' column substitutes 1.0 for any knockout
      results that did not detect a fitness effect, under the assumption that
      the knockout result is on the threshold of sensitivity.
    """

    records = []
    for site in np.flatnonzero(~skeleton.astype(bool)):
        knockout = skeleton.copy()
        knockout[site] = 1
        knockout_result = test_knockout(knockout.astype(bool))
        records.append(
            {
                "jackknife dose": knockout.astype(bool).sum(),
                # if knockout is not sensitive,
                # then assume on threshold of sensitivity
                "raw jackknife result": knockout_result,
                "jackknife result": knockout_result or 1.0,
                "site": site,
            },
        )
    return pd.DataFrame.from_records(records)
