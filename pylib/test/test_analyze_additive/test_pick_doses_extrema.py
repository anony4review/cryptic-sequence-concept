import numpy as np

from pylib.analyze_additive import pick_doses_extrema


def test_knockout_all_effective():
    num_sites = 10
    max_doses = 5
    result = pick_doses_extrema(lambda x: True, num_sites, max_doses)
    assert len(result) == max_doses
    assert np.array_equal(
        result, np.linspace(1, num_sites, max_doses, dtype=int)
    )


def test_knockout_all_ineffective():
    num_sites = 10
    max_doses = 5
    result = pick_doses_extrema(lambda x: False, num_sites, max_doses)
    assert len(result) == max_doses
    assert np.array_equal(
        result, np.linspace(1, num_sites, max_doses, dtype=int)
    )


def test_knockout_effective_range1():
    def mock_test_knockout(sample):
        return sample.sum() == 4 or sample.sum() >= 12

    num_sites = 22
    max_doses = 5
    result = pick_doses_extrema(mock_test_knockout, num_sites, max_doses)
    assert len(result) == max_doses
    assert min(result) >= 4 and max(result) <= 12


def test_knockout_effective_range2():
    def mock_test_knockout(sample):
        return sample.sum() == 4

    num_sites = 22
    max_doses = 5
    result = pick_doses_extrema(mock_test_knockout, num_sites, max_doses)
    assert len(result) == max_doses


def test_knockout_effective_range_spread():
    def mock_test_knockout(sample):
        return 4 <= sample.sum()

    num_sites = 22
    max_doses = 5
    result = pick_doses_extrema(mock_test_knockout, num_sites, max_doses)
    assert len(result) == max_doses
    assert min(result) >= 6 - 5 and max(result) <= 6 + 5
