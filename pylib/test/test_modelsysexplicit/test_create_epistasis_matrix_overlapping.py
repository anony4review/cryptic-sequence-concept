import numpy as np

from pylib.modelsys_explicit import create_epistasis_matrix_overlapping


def test_create_epistasis_matrix_overlapping():
    num_sites = 10
    num_epistatic_sets = 3
    epistatic_set_size = 10

    # Basic output structure test
    matrix = create_epistasis_matrix_overlapping(
        num_sites, num_epistatic_sets, epistatic_set_size
    )
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape[1] == num_sites
    assert matrix.shape[0] >= 3

    # Check for set creation
    unique, counts = np.unique(matrix[matrix > 0], return_counts=True)
    assert len(unique) == num_epistatic_sets
    assert all(count == epistatic_set_size for count in counts)


def test_create_epistasis_matrix_empty():
    # no sites
    matrix = create_epistasis_matrix_overlapping(
        num_sites=0, num_epistatic_sets=0, epistatic_set_size=1
    )
    assert isinstance(matrix, np.ndarray)
    assert matrix.size == 0

    # no sets
    matrix = create_epistasis_matrix_overlapping(
        num_sites=1, num_epistatic_sets=0, epistatic_set_size=1
    )
    assert isinstance(matrix, np.ndarray)
    assert matrix.sum() == 0

    # empty sets
    matrix = create_epistasis_matrix_overlapping(
        num_sites=1, num_epistatic_sets=1, epistatic_set_size=0
    )
    assert isinstance(matrix, np.ndarray)
    assert matrix.sum() == 0
