import numpy as np
import pytest

from pylib.modelsys_explicit import GenomeExplicit


def test_GenomeExplicit_initialization():
    def effect_functor(knockout: np.array) -> float:
        return np.sum(knockout)

    def artifact_functor(effect: float) -> float:
        return effect * 2

    genome = GenomeExplicit([effect_functor], [artifact_functor])
    assert len(genome._knockout_effect_functors) == 1
    assert len(genome._assay_artifact_functors) == 1


@pytest.mark.parametrize("effect_size", [1.0, 1.1, 100.1])
def test_GenomeExplicit_test_knockout_detectably_deleterious(
    effect_size: float,
):
    def effect_functor(knockout: np.array) -> float:
        return effect_size

    genome = GenomeExplicit([effect_functor])

    knockout_array = np.array([1, 1, 0, 0], dtype=bool)
    result = genome.test_knockout(knockout_array)
    assert result >= 1


@pytest.mark.parametrize("effect_size", [-1.0, -1.1, -100.1])
def test_GenomeExplicit_test_knockout_detectably_adaptive(effect_size: float):
    def effect_functor(knockout: np.array) -> float:
        return effect_size

    genome = GenomeExplicit([effect_functor])

    knockout_array = np.array([1, 1, 0, 0], dtype=bool)
    result = genome.test_knockout(knockout_array)
    assert result <= -1


def test_GenomeExplicit_test_knockout_undetectably_deleterious():
    def effect_functor(knockout: np.array) -> float:
        return 0.8

    genome = GenomeExplicit([effect_functor])

    knockout_array = np.array([1, 1, 0, 0], dtype=bool)
    result = genome.test_knockout(knockout_array)
    assert result == 0


def test_GenomeExplicit_test_knockout_neutral():
    def effect_functor(knockout: np.array) -> float:
        return 0

    genome = GenomeExplicit([effect_functor])

    knockout_array = np.array([1, 1, 0, 0], dtype=bool)
    result = genome.test_knockout(knockout_array)
    assert result == 0


def test_GenomeExplicit_test_artifact():
    def effect_functor(knockout: np.array) -> float:
        return 1

    def artifact_functor(effect: np.array) -> float:
        return 0.5

    genome = GenomeExplicit([effect_functor], [artifact_functor])

    knockout_array = np.array([1, 1, 0, 0], dtype=bool)
    result = genome.test_knockout(knockout_array)
    assert result == 0
