import typing

import numpy as np
import opytional as opyt
from scipy import stats as scipy_stats

from ..auxlib._nonzero import nonzero
from ..auxlib._nunique import nunique


class CalcKnockoutEffectsEpistasis:
    """Functor to model the contribution of epistatic interactions between
    genome sites to knockout outcomes."""

    _effect_thresh: int
    _effect_size: np.array
    _epistasis_matrix: np.array

    def __init__(
        self: "CalcKnockoutEffectsEpistasis",
        epistasis_matrix: np.array,
        effect_thresh: typing.Optional[int] = None,
        effect_size: typing.Union[float, typing.Tuple[float, float]] = 1.0,
    ) -> None:
        """Initialize functor with epistatic interaction information.

        Parameters
        ----------
        epistasis_matrix : np.array
            Arrangement of epistatic set labels, where columns correspond to
            genome sites and rows allow each site to contain more than one
            epistatic value. Only nonzero values are considered as set labels.

            Can be generated via `create_epistasis_matrix_overlapping` or
            `create_epistasis_matrix_disjoint`.

        effect_thresh : int, optional
            The threshold number of activations required to consider an effect.
            If None, uses the largest set size in the epistatic matrix.

        effect_size : Union[float, Tuple[float, float]], default=1.0
            The size of fitness effect when threshold sites within an epistatic
            set are knocked out.

            If a tuple, interpreted as a lower and upper bound for a uniform
            distribution from which to draw effect sizes.
        """
        self._effect_thresh = opyt.or_value(
            effect_thresh,
            scipy_stats.mode(
                nonzero(epistasis_matrix), axis=None, keepdims=False
            ).count,
        )
        num_epistasis_sets = nunique(epistasis_matrix[epistasis_matrix != 0])
        try:
            lb, ub = effect_size
            self._effect_size = np.random.uniform(
                lb, ub, size=num_epistasis_sets
            )
        except TypeError:
            self._effect_size = np.full(num_epistasis_sets, effect_size)

        self._epistasis_matrix = epistasis_matrix

    def __call__(
        self: "CalcKnockoutEffectsEpistasis",
        knockout: np.array,
    ) -> float:
        """Calculate the fitness effect of a knockout considering epistatic
        interactions.

        Parameters
        ----------
        knockout : np.array
            A binary array representing knockout sites, where 1 indicates site
            knockout. Length of array corresponds to genome size.

        Returns
        -------
        float
            The fitness effect of knockouts from epistatic interactions.

            If the number of knocked out sites within an epistatic set exceeds
            `effect_thresh`, returns `effect_size`; otherwise, returns 0.0.
        """
        active_sites = self._epistasis_matrix * knockout
        values, counts = np.unique(active_sites, return_counts=True)
        activations = np.zeros_like(self._effect_size)
        if values.size and values[-1:] > 0:
            has_leading_zero = not values[0]
            counts_ = counts[has_leading_zero:]
            values_ = values[has_leading_zero:] - 1
            activations[values_] = counts_ >= self._effect_thresh

        return (activations * self._effect_size).sum()
