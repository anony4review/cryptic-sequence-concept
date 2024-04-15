from numbers import Integral

import numpy as np
from scipy import stats as scipy_stats

from ..auxlib._cumcount import cumcount


def create_epistasis_matrix_overlapping(
    num_sites: int, num_epistatic_sets: int, epistatic_set_size: int
):
    """Generate a matrix specifying epistatic interactions between genome
    sites, allowing each site to have more than one epistatic interaction.

    Epistatic redundancy is modeled as sets of n sites tagged with a common
    epistatic set label.

    Parameters
    ----------
    num_sites : int
        Number of genome sties
    num_epistatic_sets : int
        The number of epistatic sets to be created.
    epistatic_set_size : int
        The size of each epistatic set.

    Returns
    -------
    numpy.ndarray
        Arrangement of epistatic redundancy labels, where columns correspond to
        genome sites and rows allow each site to contain more than one epistatic
        value.

    Notes
    -----
    - Epistatic set labels are indexed starting from 1 to distinguish from
      empty zeros.
    - The height of the matrix is determined by the site with the most
      epistatic labels.

    See Also
    --------
    create_epistasis_matrix_disjoint : Create an epistasis matrix without
    any site-wise overlap between epistatic sets.
    CalcKnockoutEffectsEpistasis : Uses epistasis matrix to calculate the
    epistatic effects of a knockout.
    """
    # assign each set value to a random site...
    # more than one set value may be assigned per site
    site_indices = np.empty((num_epistatic_sets, epistatic_set_size), dtype=int)
    for i in range(num_epistatic_sets):
        site_indices[i] = np.random.choice(
            num_sites, epistatic_set_size, replace=False
        )  # replace=False: no vals from same set at same site

    # how many rows are necessary for the site with most epistatic values?
    nrows = (
        scipy_stats.mode(site_indices, axis=None, keepdims=False).count
        if num_epistatic_sets * epistatic_set_size  # empty case
        else 1
    )
    assert isinstance(nrows, Integral)

    # corresponding to that set's identity
    epistasis_values = (
        np.tile(np.arange(num_epistatic_sets), (epistatic_set_size, 1)).ravel()
        + 1  # set ids are 1-indexed to distinguish from empty zero
    )
    col_indices = site_indices.T.ravel()  # col indices are the sites
    # row position corresponds to number preceding identical values
    row_indices = cumcount(site_indices.T.ravel())

    # organize epistasis values by site
    epistasis_matrix = np.zeros((nrows, num_sites), dtype=int)
    epistasis_matrix[row_indices, col_indices] = epistasis_values

    return epistasis_matrix
