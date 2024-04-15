import numpy as np
import pandas as pd

from pylib.analyze_agnostic import assay_agnostic_naive
from pylib.analyze_epistasis import describe_skeletons, skeletonize_naive
from pylib.modelsys_explicit import (
    CalcKnockoutEffectsAdditive,
    CalcKnockoutEffectsEpistasis,
    GenomeExplicit,
    create_additive_array,
    create_epistasis_matrix_disjoint,
    describe_additive_array,
    describe_epistasis_matrix,
)


def test_assay_agnostic_naive_smoke():
    num_sites = 1000
    distn = lambda x: np.random.rand(x) * 0.7  # noqa: E731
    additive_array = create_additive_array(num_sites, 0.04, distn)
    epistasis_matrix = create_epistasis_matrix_disjoint(num_sites, 40, 4)
    genome = GenomeExplicit(
        [
            CalcKnockoutEffectsAdditive(additive_array),
            CalcKnockoutEffectsEpistasis(
                epistasis_matrix, effect_size=(0.7, 1.6)
            ),
        ],
    )

    dfa = describe_additive_array(additive_array)
    dfb = describe_epistasis_matrix(epistasis_matrix)
    df_genome = pd.DataFrame.merge(dfa, dfb, on="site")
    df_genome["site type"] = (
        df_genome["additive site"].astype(int)
        + df_genome["epistasis site"].astype(int) * 2
    ).map(
        {
            0: "neutral",
            1: "additive",
            2: "epistasis",
            3: "both",
        }
    )

    num_skeletonizations = 5
    skeletons = np.vstack(
        [
            skeletonize_naive(num_sites, genome.test_knockout)
            for _ in range(num_skeletonizations)
        ],
    )

    df_skeletons = describe_skeletons(skeletons, genome.test_knockout)

    res = assay_agnostic_naive(df_skeletons)
    assert isinstance(res, dict)
