from pylib.analyze_additive import assay_additive_naive, pick_doses_extrema
from pylib.modelsys_explicit import (
    CalcKnockoutEffectsAdditive,
    GenomeExplicit,
    create_additive_array,
)


def test_assay_additive_naive_smoke():
    num_sites = 1000
    additive_array = create_additive_array(num_sites, 0.05)  # 50 sites
    genome = GenomeExplicit(
        [CalcKnockoutEffectsAdditive(additive_array)],
    )
    knockout_doses = pick_doses_extrema(
        genome.test_knockout, num_sites, max_doses=5
    )
    est = assay_additive_naive(
        genome.test_knockout, num_sites, knockout_doses, num_replications=1000
    )
    assert isinstance(est, dict)
