import numpy as np

from pylib.analyze_epistasis import skeletonize_naive


def test_skeletonize_naive_knockout_all_sites_sensitive():
    def mock_test_knockout(knockout: np.array) -> bool:
        return np.any(knockout)  # sensitive to any knockout

    num_sites = 10
    skeleton = skeletonize_naive(num_sites, mock_test_knockout)

    assert len(skeleton) == num_sites  # correct length
    assert not np.any(skeleton)  # no sites should be knocked out
    assert set(skeleton) == set(
        # exclude zero if all sites nopped out
        [*range(skeleton.astype(bool).sum() + 1)][-num_sites:]
    )  # all sites in ordering


def test_skeletonize_naive_knockout_effect_no_sites_sensitive():
    def mock_test_knockout(knockout: np.array) -> bool:
        return False  # sensitive to no knockouts

    num_sites = 10
    skeleton = skeletonize_naive(num_sites, mock_test_knockout)

    assert len(skeleton) == num_sites  # correct length
    assert np.all(skeleton)  # all sites should be knocked out
    assert set(skeleton) == set(
        # exclude zero if all sites nopped out
        [*range(skeleton.astype(bool).sum() + 1)][-num_sites:]
    )  # all sites in ordering


def test_skeletonize_naive_single_sensitive_site():
    def mock_test_knockout(knockout):
        return knockout[0]  # sensitive to only the first site

    expected = np.array([0, 1, 1])
    skeleton = skeletonize_naive(expected.size, mock_test_knockout)
    assert np.array_equal(skeleton.astype(bool), expected)
    assert set(skeleton) == set(
        range(skeleton.astype(bool).sum() + 1)
    )  # all sites in ordering
