from scipy.special import logsumexp
from itertools import combinations_with_replacement, product
from .joint_distribution import create_combined_matrices
import numpy as np
np.set_printoptions(precision=2, suppress=True)

MAIN = -1
HELPER = -2
M = MAIN
H = HELPER


def _get_masked_calc_func(k_r, k_a, threshold):
    mask = np.where((k_r>threshold) & (k_a>threshold), -np.inf, 0)
    print("MASKED", np.sum(mask==-np.inf))
    def calc_func_masked(count_matrix, offset):
        m = mask[:-offset] if offset>0 else mask[-offset:]
        l = calc_likelihood(count_matrix)
        assert m.shape==l.shape, (m.shape, l.shape)
        return l+m
    return calc_func_masked


def calc_likelihood(count_matrix):
    count_matrix = count_matrix+1
    p = count_matrix/count_matrix.sum(axis=M, keepdims=True)
    return np.sum(count_matrix*np.log(p), axis=(M, H))/np.sum(count_matrix, axis=(M, H))
    
def calc_argmax(count_matrix):
    return np.sum(np.max(count_matrix, axis=M), axis=-1)/count_matrix.sum(axis=(M, H))


def find_best_helper(combined, score_func, N, with_model=False):
    best_idx, best_score = np.empty(N, dtype="int"), -np.inf*np.ones(N)
    for j, counts in enumerate(combined, 1):
        if j % 100 == 0:
            print(f"Best helper offset: {j}")
        scores = score_func(counts, j) if with_model else score_func(counts)
        do_update = scores > best_score[j:]
        best_score[j:][do_update] = scores[do_update]
        best_idx[j:][do_update] = np.flatnonzero(do_update)
        rev_scores = score_func(counts.swapaxes(-2, -1), -j) if with_model else score_func(counts.swapaxes(-2, -1))
        do_update = rev_scores>best_score[:-j]
        best_score[:-j][do_update] = rev_scores[do_update]
        best_idx[:-j][do_update] = np.flatnonzero(do_update)+j
    assert np.all(best_idx<N), np.where(best_idx>N)
    return best_idx

class HelperModel:
    def __init__(self, model, helper_variants, genotype_probs):
        self._model = model
        self._helper_variants = np.asanyarray(helper_variants)
        self._genotype_probs = np.asanyarray(genotype_probs) # np.log(genotype_combo_matrix/genotype_combo_matrix.sum(axis=-1, keepdims=True))

    def predict(self, k1, k2):
        scores = self.score(k1, k2)
        return np.argmax(scores, axis=-1)
        probs = [self.logpmf(k1, k2, g) for g in range(3)]
        return np.argmax(probs, axis=0)

    def score(self, k1, k2):
        count_probs = np.array([self._model.logpmf(k1, k2, g) for g in [0, 1, 2]]).T
        log_probs =  self._genotype_probs + count_probs[self._helper_variants].reshape(-1, 3, 1)+count_probs.reshape(-1, 1, 3)
        unnormalized = logsumexp(log_probs, axis=H)
        return unnormalized-logsumexp(unnormalized, axis=-1, keepdims=True)

    @classmethod
    def from_genotype_matrix(cls, model, genotype_matrix, score_func=calc_likelihood, with_model=False, dummy_counts=1, window_size=20):
        
        combined = create_combined_matrices(genotype_matrix, window_size)
        helpers = find_best_helper(combined, score_func, len(genotype_matrix), with_model)
        # helpers = find_best_helper(combined, calc_argmax)
        helper_counts = genotype_matrix[helpers]*3
        flat_idx = genotype_matrix+helper_counts
        genotype_combo_matrix = np.array([(flat_idx==k).sum(axis=-1) for k in range(9)]).T.reshape(-1, 3, 3)+dummy_counts
        genotype_probs = np.log(genotype_combo_matrix/genotype_combo_matrix.sum(axis=(-1, -2), keepdims=True))
        return cls(model, helpers, genotype_probs)

    def logpmf(self, ref_counts, alt_counts, genotype):
        count_probs = np.array([self._model.logpmf(ref_counts, alt_counts, g) for g in [0, 1, 2]]).T
        log_probs =  self._genotype_probs+count_probs[self._helper_variants].reshape(-1, 3, 1)+count_probs.reshape(-1, 1, 3)
        return logsumexp(log_probs, axis=H)[..., genotype]

    def diagnostics(self, idx):
        diag = {}
        diag["helper"] = self._helper_variants[idx]
        diag["genotype_probs"] = self._genotype_probs[idx]
        diag.update(self._model.diagnostics(idx))
        return diag
class SimpleHelperModel(HelperModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._m_given_h = self._genotype_matrix/np.logsumexp(self._genotype_matrix)

    def _get_helper_genotypes(self, count_probs):
        probs = count_probs[self._helper_variants]+logsumexp(self._genotype_probs[self._helper_variants], axis=M)
        return np.argmax(probs, axis=-1)

    def score(self, k1, k2):
        count_probs = np.array([self._model.logpmf(k1, k2, g) for g in [0, 1, 2]]).T
        helper_g = self._get_helper_genotypes(count_probs)
        genotype_probs = self._genotype_probs[np.arange(k1.shape[0]), helper_g, :]
        return count_probs+genotype_probs
        
    def logpmf(self, ref_counts, alt_counts, genotype):
        count_probs = np.array([self._model.logpmf(ref_counts, alt_counts, g) for g in [0, 1, 2]]).T
        helper_g = self._get_helper_genotypes(count_probs)
        res = count_probs[:, genotype] + np.array([self._genotype_probs[i, h, genotype] for i, h in  enumerate(helper_g)]) # self._genotype_probs[:, helper_g, genotype]
        assert res.shape == (self._genotype_probs.shape[0],), res.shape
        return res

class PriorModel:
    def __init__(self, model, genotype_probs):
        self._model = model
        self._n_variants = self._model._n_variants

        self._genotype_frequencies = genotype_probs # 
        assert np.allclose(logsumexp(self._genotype_frequencies, axis=-1), 0)

    def predict(self, k1, k2):
        probs = [self.logpmf(k1, k2, g) for g in range(3)]
        return np.argmax(probs, axis=0)

    def score(self, k1, k2):
        unnormalized = np.array([self.logpmf(k1, k2, g) for g in [0, 1, 2]]).T
        return unnormalized-logsumexp(unnormalized, axis=-1, keepdims=True)

    @classmethod
    def from_genotype_matrix(cls, model, genotype_matrix):
        genotype_counts = np.array([np.sum(genotype_matrix==k, axis=-1) for k in range(3)]).T+0.1
        genotype_probs = np.log(genotype_counts/genotype_counts.sum(axis=-1, keepdims=True))
        return cls(model, genotype_probs)

    def logpmf(self, ref_counts, alt_counts, genotype):
        return self._model.logpmf(ref_counts, alt_counts, genotype)+self._genotype_frequencies[:, genotype]


"""
sum_(g_H) P(G_H)*P(g_H|G_H)*
"""
def calc_likelihood_for(count_matrix, prob_dists):
    """
    count_matrix = [g_h,g_m]
    p[g_h,k] = prob of observing k of total_reads on helper ref if gneotype is g_h on helper variant
    """
    N = prob_dists.shape[-1]-1
    t = 0 
    M  = count_matrix
    p = prob_dists
    for g_h in range(3):
        for g_m in range(3):
            for k in range(N+1):
                t += M[g_h, g_m]*p[g_h, k]*np.log(sum([p[x, k]*M[x, g_m]/np.sum(p[:, k]) for x in range(3)]))
    return t

def get_scores(count_matrices, prob_dists):
    return np.array([calc_likelihood_for(count_matrix, prob_dist)
                     for count_matrix, prob_dist in zip(count_matrices, prob_dists)])
        

def full_solution(combined, prob_dists):
    """
    combined: (w, n-1->n-w, 3, 3)
    prob_dists: (n, 3, total_reads)
    p[v,g,k] = prob of observing k of total_reads on ref if gneotype ig on varaint v
    """
    N = len(combined[0])+1
    best_idx, best_score = np.empty(N), -np.inf*np.ones(N)
    for j, counts in enumerate(combined, 1):
        
        scores = get_scores(counts, prob_dists[:-j])
        do_update = scores>best_score[j:]
        best_score[j:][do_update] = scores[do_update]
        best_idx[j:][do_update] = np.flatnonzero(do_update)
        rev_scores = get_scores(counts.swapaxes(-2, -1), prob_dists[j:])
        do_update = rev_scores>best_score[:-j]
        best_score[:-j][do_update] = rev_scores[do_update]
        best_idx[:-j][do_update] = np.flatnonzero(do_update)+j
    return best_idx

def simulate_prob_dists(n_variants, N):
    ps = np.random.rand(n_variants, 3, N)
    return ps/ps.sum(axis=-1, keepdims=True)
