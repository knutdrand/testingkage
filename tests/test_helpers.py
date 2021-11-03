import pytest
from kmer_align.helper_variants import find_best_helper, create_combined_matrices, calc_likelihood, SimpleHelperModel
import numpy as np

@pytest.fixture()
def genotype_matrix():
    return [[0, 1, 2],
            [0, 1, 1],
            [0, 0, 1],
            [0, 0, 1],
            [1, 2, 2],
            [2, 1, 0]]

@pytest.fixture()
def small_genotype_matrix():
    return [[0, 1, 2],
            [0, 1, 1],
            [0, 0, 1]]

@pytest.fixture()
def tricky_genotype_matrix():
    return [[1, 1, 1, 0, 0],
            [0, 2, 2, 2, 2],
            [1, 1, 1, 1, 1]]


def test_combined_matrices(small_genotype_matrix):
    combined = create_combined_matrices(small_genotype_matrix, 2)
    r = [[[[1, 0, 0],
           [0, 1, 0],
           [0, 1, 0]],
          [[1, 0, 0],
           [1, 1, 0],
           [0, 0, 0]]],
         [[[1, 0, 0],
           [1, 0, 1],
           [0, 1, 0]]]]
          
    for p, t in zip(combined, r):
        assert np.all(np.array(t)==p)

def test_find_best_helper(genotype_matrix):
    combined = create_combined_matrices(genotype_matrix, 6)
    helpers = find_best_helper(combined, calc_likelihood)
    assert np.all(helpers == [5, 4, 3, 2, 1, 0])

def test_tricky_find_best_helper(tricky_genotype_matrix):
    combined = create_combined_matrices(tricky_genotype_matrix, 6)
    helpers = find_best_helper(combined, calc_likelihood)
    assert np.all(helpers == [2, 2, 1])
                 

def test_simple_helper_model():
    probs = [[[0.8, 0.15, 0.05],
              [0.3, 0.3, 0.4],
              [0.5, 0.25, 0.25]],
             [[0.2, 0.6, 0.2],
              [0.3, 0.3, 0.4],
              [0.5, 0.25, 0.25]]]
    model = SimpleHelperModel(DummyModel, [1, 0], np.log(probs))
    pmfs = np.exp([model.logpmf([0, 0], [0, 0], g) for g in (0, 1, 2)]).T
    true = np.array([[0.8*0.5, 0.15*0.25, 0.05*0.25],
                     [0.2*0.5, 0.6*0.25, 0.2*0.25]])
    assert np.allclose(pmfs/pmfs.sum(axis=-1, keepdims=True), true/true.sum(axis=-1, keepdims=True))
      
class DummyModel:
    @classmethod
    def logpmf(cls, alt_counts, ref_counts, genotype):
        return np.log(0.25+0.25*(genotype==0))*np.ones_like(alt_counts)

