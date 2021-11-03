import numpy as np
from scipy.stats import binom
from scipy.special import logsumexp

MAIN = -1
HELPER = -2
M = MAIN
H = HELPER


class BinomialModel:

    error_rate = 0.01
    def __init__(self, expected_ref, expected_alt):
        self._expected_ref = 2*np.asanyarray(expected_ref)
        self._expected_alt = 2*np.asanyarray(expected_alt)
        self._n_variants = len(self._expected_ref)

    def predict(self, k1, k2):
        return np.argmax([self.logpmf(k1, k2, g) for g in (0, 1, 2)], axis=0)

    def score(self, k1, k2):
        ps = np.array([self.logpmf(k1, k2, g) for g in (0,1,2)])
        return ps-logsumexp(ps, axis=0, keepdims=True)

    def logpmf(self, k1, k2, genotype):
        k1 = np.asanyarray(k1)
        k2 = np.asanyarray(k2)
        total = k1+k2
        ps = (self._expected_ref+2-genotype+self.error_rate)/(2+self._expected_ref+self._expected_alt + 2*self.error_rate)
        return binom.logpmf(k1, total, ps)

def get_helper_to_predicted_helper_probs(model, N):
    transition_matrix = np.zeros((3, 3))
    for G_h in (0, 1, 2):
        for k in range(N+1):
            p_k = np.exp(model.logpmf(k, N-k, G_h))
            pred_G_h = model.predict(k, N-k)
            transition_matrix[G_h, pred_G_h] += p_k
    return transition_matrix

def np_get_helper_to_predicted_helper_probs(model, N):
    transition_matrix = np.zeros((model._n_variants, 3, 3))
    k = np.arange(N+1)[:, None]
    for G_h in (0, 1, 2):
        p_k = np.exp(model.logpmf(k, N-k, G_h))
        predicted = model.predict(k, N-k)
        for pred_G_h in (0, 1, 2):
            transition_matrix[:, G_h, pred_G_h] = np.sum((predicted==pred_G_h)*p_k, axis=0)
    # for t, r, a in zip(transition_matrix, model._expected_ref, model._expected_alt):
    #    print(r, a, t.trace(), np.log(t).trace())
    return transition_matrix

def get_internal_helper_scores(model, genotype_frequencies):
    k = np.arange(N+1)[:, None]
    for G_h in (0, 1, 2):
        p_k = np.exp(model.logpmf(k, N-k, G_h))
        scores = model.score(k, N-k)
        
        for pred_G_h in (0, 1, 2):
            transition_matrix[:, G_h, pred_G_h] = np.sum((predicted==pred_G_h)*p_k, axis=0)

def _np_get_helper_to_predicted_helper_probs(model, N):
    transition_matrix = np.zeros((model._n_variants, 3, 3))
    k = np.arange(N+1)[:, None]
    for G_h in (0, 1, 2):
        p_k = np.exp(model.logpmf(k, N-k, G_h))
        scores = model.score(k, N-k)
        transition_matrix[:, G_h, :] = (p_k[None, ...]*np.exp(scores)).sum(axis=1).T
        #predicted = model.predict(k, N-k)
        #for pred_G_h in (0, 1, 2):
        #     transition_matrix[:, G_h, pred_G_h] = np.sum((predicted==pred_G_h)*p_k, axis=0)
    return transition_matrix


def get_calc_func(transition_matrix):
    transition_matrix = np.asanyarray(transition_matrix)
    print(np.mean(transition_matrix, axis=0))
    def calc_likelihood_with_model(count_matrix, offset):
        assert offset != 0
        if offset > 0:
            T = transition_matrix[:-offset]
        else:
            T = transition_matrix[-offset:]
        assert count_matrix.shape==T.shape, (count_matrix.shape, T.shape)
        # C = count_matrix
        # p_g_h = C.sum(axis=-1, keepdims=True)/C.sum(axis=(-1, -2), keepdims=True)
        V = np.einsum("vkj,vki->vij", count_matrix, T)
        #print(V.sum(), count_matrix.sum())
        count_matrix = count_matrix+1
        p = count_matrix/count_matrix.sum(axis=M, keepdims=True)
        return np.sum(count_matrix*np.log(p)+np.log((T+0.01).diagonal(axis1=-2, axis2=-1))[..., None], axis=(M, H))
    return calc_likelihood_with_model


def get_calc_func_with_model(k_r, k_a, N):
    # print(r_k, r_a)
    model = BinomialModel(k_r, k_a)
    T = np_get_helper_to_predicted_helper_probs(model, N)
    # print(T)
    return get_calc_func(T)

if __name__ == "__main__":
    model = BinomialModel(1, 1)
    t = get_helper_to_predicted_helper_probs(model, 10)
    np.set_printoptions(precision=3, suppress=True)
    
    
    print(t)
    print(t.sum(axis=-1))
    np_model = BinomialModel([0, 1], [1, 1])
    print(np_get_helper_to_predicted_helper_probs(np_model, 10))
