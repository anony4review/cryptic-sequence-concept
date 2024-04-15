import numpy as np


class CalcKnockoutEffectsAdditive:
    """Functor to model contribution of additive small-effect genome sites.
    to knockout outcome."""

    _additive_array: np.array

    def __init__(
        self: "CalcKnockoutEffectsAdditive", additive_array: np.array
    ) -> None:
        """Initialize functor with site effect information.

        Parameters
        ----------
        additive_array : np.array
            A one-dimensional array with size equal to genome size, with each
            entry representing the fitness effect of knocking a corresponding
            genome site out.

            Perfectly neutral sites have value zero. Adaptive sites have
            positive values and maladaptive sites, if any, would have negative
            values. Can be generated e.g., via `create_additive_array`.
        """
        self._additive_array = additive_array

    def __call__(
        self: "CalcKnockoutEffectsAdditive", knockout: np.array
    ) -> float:
        """Calculate the fitness effect of a knockout.

        Parameters
        ----------
        knockout : np.array
            A binary array representing knockout sites, where 1 indicates site
            knockout.

            Length of array corresponds to genome size, i.e., one entry per
            genome site.

        Returns
        -------
        float
            The cumulative effect of the knockouts, calculated as the sum of
            contributions from the knocked out sites.

            Positive values indicate a deleterious knockout effect, with 1.0
            taken as the detectability threshold (by convention elsewhere).
        """
        active = self._additive_array * knockout
        return np.sum(active)
