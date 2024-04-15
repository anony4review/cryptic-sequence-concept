import typing

import numpy as np

from ..auxlib._discrete_geomspace import discrete_geomspace
from ._sample_knockout import sample_knockout


def pick_doses_extrema(
    test_knockout: typing.Callable,
    num_sites: int,
    max_doses: int,
    smear_count: int = 100,
) -> np.array:
    """Pick dose levels spanning a genome's sensitivity to knockouts, from the
    smallest dose with a fitness effect through the largest dose without a
    fitness effect.

    Parameters
    ----------
    test_knockout : typing.Callable
        A function that tests the effect of a knockout, e.g., `GenomeExplicit.test_knockout`.
    num_sites : int
        Number of sites in genome.
    max_doses : int
        Number of doses to pick, given sufficient sties available.
    smear_count : int, optional
        The number of doses to be considered in the discrete geometric space,
        default is 100.

    Returns
    -------
    np.array
        An array of unique knockout counts, in ascending order.
    """
    smear_doses = discrete_geomspace(1, num_sites, smear_count, dtype=int)
    smear_results = np.array(
        [
            test_knockout(sample_knockout(dose, num_sites))
            for dose in smear_doses
        ],
    ).astype(bool)
    if np.all(smear_results) or not np.any(smear_results):
        return np.unique(np.linspace(1, num_sites, max_doses, dtype=int))

    first_affected = smear_doses[np.flatnonzero(smear_results)[0]]
    last_unaffected = smear_doses[np.flatnonzero(~smear_results)[-1]]

    # expand to available space
    while last_unaffected - first_affected < max_doses and (
        last_unaffected < num_sites or first_affected > 0
    ):
        last_unaffected += last_unaffected < num_sites
        first_affected -= (
            first_affected > 0 and last_unaffected - first_affected < max_doses
        )

    return np.unique(
        np.linspace(first_affected, last_unaffected, max_doses, dtype=int),
    )
