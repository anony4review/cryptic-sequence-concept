import typing

import numpy as np


class GenomeExplicit:
    """A genome with explicitly-specified knockout fitness effects and
    configurable fitness assay limitations on or distortions of detectability.
    """

    _assay_artifact_functors: typing.List[typing.Callable]
    _knockout_effect_functors: typing.List[typing.Callable]

    def __init__(
        self: "GenomeExplicit",
        knockout_effects: typing.Sequence[typing.Callable],
        assay_artifacts: typing.Sequence[typing.Callable] = tuple(),
    ) -> None:
        """Configure fitness response to knockout and characteristics of assay
        to detect fitness effects.

        Parameters
        ----------
        knockout_effects : typing.Sequence[typing.Callable]
            A sequence of functions, each modeling a different knockout effect
            on the genome.

            Knockout effect functions should take a knockout a binary mask
            representing knockout sites, where 1 indicates site knockout.
            Functions should return float value indicating fitness effect,
            with deleterious effects positive, advantageous effects negative,
            and neutral effects zero.


            Net fitness effect is calculated as the sum of individual effects'
            results. Fitness effects are considered detectable only with
            absolute value greater than or equal to 1.
        assay_artifacts : typing.Sequence[typing.Callable], optional
            A sequence of functions representing various assay artifacts, by
            default an empty tuple.

        See Also
        --------
        CalcKnockoutEffectsAdditive, CalcKnockoutEffectsEpistasis
        """
        self._knockout_effect_functors = [*knockout_effects]
        self._assay_artifact_functors = [*assay_artifacts]

    def test_knockout(self: "GenomeExplicit", knockout: np.array) -> float:
        """Test sign of knockout fitness effect, if observable.

        Calculates knockout effect as sum of effects from knockout effect
        functors after sequenced through assay artifact functors.

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
            Value >=1.0 if the cumulative effect of the knockouts is detectably
            deleterious, =<-1.0 if cumulative effects of knockouts is detectably
            adaptive, 0 otherwise.
        """
        if knockout.size and knockout.dtype != bool:
            raise ValueError(
                f"Knockout array should be boolean, but was {knockout.dtype}.",
            )
        result = sum(
            effect(knockout) for effect in self._knockout_effect_functors
        )
        for artifact in self._assay_artifact_functors:
            result = artifact(result)
        # TODO refactor: make thresholding an assay artifact instead of hardcode
        return result * (np.abs(result) >= 1.0)
