import pytest

from pylib.analyze_epistasis import describe_skeletons, skeletonize_naive
from pylib.modelsys_explicit import (
    CalcKnockoutEffectsAdditive,
    CalcKnockoutEffectsEpistasis,
    GenomeExplicit,
    create_additive_array,
    create_epistasis_matrix_overlapping,
)


@pytest.mark.parametrize("num_skeletons", [0, 1, 10])
def test_describe_skeletons(num_skeletons: int):
    num_sites = 100
    additive_array = create_additive_array(num_sites, 0.1)
    epistasis_matrix = create_epistasis_matrix_overlapping(num_sites, 4, 2)
    genome = GenomeExplicit(
        [
            CalcKnockoutEffectsAdditive(additive_array),
            CalcKnockoutEffectsEpistasis(epistasis_matrix),
        ],
    )

    skeletons = [
        skeletonize_naive(num_sites, genome.test_knockout)
        for _ in range(num_skeletons)
    ]
    res = describe_skeletons(skeletons, genome.test_knockout)

    if num_skeletons:
        assert set(res["site"]) == set(range(num_sites))
