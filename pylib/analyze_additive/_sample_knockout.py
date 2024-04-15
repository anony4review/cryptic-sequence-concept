import numpy as np


def sample_knockout(dose: int, num_sites: int) -> np.array:
    """Sample a knockout configuration with `dose` sites sampled uniformly over
    `num_sites` genome positions.

    Returns
    -------
    np.array
        A binary mask where 1 represents a knockout at a site and 0 represents
        no knockout.

        The length of the array is equal to `num_sites`.
    """
    res = np.zeros(num_sites, dtype=bool)
    res[np.random.choice(num_sites, dose, replace=False)] = 1
    return res
