from numbers import Integral, Number
import typing
import warnings

import numpy as np


def create_additive_array(
    num_sites: int,
    effect_prevalence: Number,
    effect_size_distribution: typing.Callable = np.random.rand,
) -> np.array:
    """Create an array representing additive genetic effects.

    Parameters
    ----------
    num_sites : int
        The number of sites (e.g., genes or loci) in the genome.
    effect_prevalence : Number
        The fraction of sites to be affected, if between 0 and 1, otherwise the
        absolute number of sites to be affected.
    effect_size_distribution : typing.Callable, default numpy.rand
        A function to generate random numbers for the magnitude of effects.

        Default np.random.rand, which generates random numbers between 0 and 1.
        Pass, e.g., `lambda x: np.random.rand(x) * y` to scale effect size.

    Returns
    -------
    np.array
        Array with knockout effects per genome site, with positive values
        deleterious, negative values advantageous, and zero values neutral.
    """
    res = np.zeros(num_sites)
    return _add_additive_effects(
        res, effect_prevalence, effect_size_distribution
    )


def _add_additive_effects(
    array: np.array,
    effect_prevalence: Number,
    effect_size_distribution: typing.Callable = np.random.rand,
) -> np.array:
    """Add additional additive effects to an existing additive effects array.

    This function is used internally by `create_additive_array` to populate an
    zero-initialized array with additive effects.

    Parameters
    ----------
    array : np.array
        A numpy array to which the effects will be added. This array should be
        initialized to zeros.
    effect_prevalence : Number
        The fraction of sites to be affected, if between 0 and 1, otherwise the
        number of sites to be affected.
    effect_size_distribution : typing.Callable, default numpy.random.rand
        A function to generate random numbers for the magnitude of effects.

        Default np.random.rand, which generates random numbers between 0 and 1.
        Pass, e.g., `lambda x: np.random.rand(x) * y` to scale effect size.

    Returns
    -------
    np.array
        Array with knockout effects per genome site, with positive values
        deleterious, negative values advantageous, and zero values neutral.
    """
    if effect_prevalence == 1:
        warnings.warn(
            "Effect prevalence interpretation is ambiguous for value one, "
            "treating as count if integral else rate.",
        )
    num_sites = array.size
    num_effects = (
        int(num_sites * effect_prevalence)
        if effect_prevalence < 1.0
        or effect_prevalence == 1.0
        and not isinstance(effect_prevalence, Integral)
        else int(effect_prevalence)
    )

    effects = effect_size_distribution(num_effects)
    indices = np.random.choice(num_sites, num_effects, replace=False)
    array[indices] += effects
    return array
