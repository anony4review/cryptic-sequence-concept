import numpy as np
from scipy import stats as scipy_stats


def create_epistasis_matrix_disjoint(
    num_sites: int, num_epistatic_sets: int, epistatic_set_size: int
) -> np.array:
    """Generate a matrix specifying epistatic interactions between genome
    sites, allowing each site to have at most one epistatic interaction.

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
        genome sites and there is only a single row.

    Notes
    -----
    - Epistatic set labels are indexed starting from 1 to distinguish from
      empty zeros.

    See Also
    --------
    create_epistasis_matrix_overlapping : Create an epistasis matrix with
    potential site-wise overlap between epistatic sets.
    CalcKnockoutEffectsEpistasis : Uses epistasis matrix to calculate the
    epistatic effects of a knockout.
    """
    # assign each set value to a random site...
    # with no two assigned to the same site
    site_indices = np.empty((num_epistatic_sets, epistatic_set_size), dtype=int)
    site_indices.flat = np.random.choice(
        num_sites, num_epistatic_sets * epistatic_set_size, replace=False
    )

    # no site should have more than one epistasis set member
    # <= 1 handles empty case
    assert 1 >= np.nan_to_num(
        scipy_stats.mode(site_indices, axis=None, keepdims=False).count,
        0,
    )

    # each epistasis set gets its own column, with identical value
    # corresponding to that set's identity (1-indexed)
    epistasis_values = (
        np.tile(np.arange(num_epistatic_sets), (epistatic_set_size, 1)).ravel()
        + 1  # set ids are 1-indexed to distinguish from empty zero
    )
    col_indices = site_indices.T.ravel()  # col indices are the sites
    row_indices = np.zeros_like(col_indices)

    # organize epistasis values by site
    epistasis_matrix = np.zeros((1, num_sites), dtype=int)
    epistasis_matrix[row_indices, col_indices] = epistasis_values

    return epistasis_matrix
