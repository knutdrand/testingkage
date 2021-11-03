import numpy as np
from scipy.special import logsumexp
from .helper_finder import BinomialModel

M, H = (-1, -2)
class FullModel:
    def __init__(self, expected_ref, expected_alt, genotype_log_probs, N=20):
        self._count_model = BinomialModel(expected_ref, expected_alt)
        self._genotype_log_probs = genotype_log_probs
        self._N = N
        self._n_variants =expected_ref.shape[0]
        self._init_calcs()

    def _init_calcs(self):
        """
        n_variants x 3 x N+1
         (V, G_h, K_h)
        """
        self._count_log_probs = np.array([[self._count_model.logpmf(k, self._N-k, G) for G in (0, 1,2)]
                                          for k in range(self._N+1)]).T
        assert np.allclose(logsumexp(self._count_log_probs, axis=-1), 0)
        self._marginal_count_probs = logsumexp(self._count_log_probs+self._genotype_log_probs[..., None], axis=1)
        assert np.allclose(logsumexp(self._marginal_count_probs, axis=-1), 0)
        # maybe simpler with P(G|k) using scores. 

    @staticmethod
    def _get_window(array, offset):
        return array[:-offset] if offset > 0 else array[-offset:]

    def full_score_func(self, count_matrix, offset):
        """N
        counts = (V, G_h, G_m)
        """
        count_matrix += 1
        log_prior = np.log(count_matrix/count_matrix.sum(axis=(-2, -1), keepdims=True))
        count_log_probs = self._count_log_probs[:-offset] if offset>0 else self._count_log_probs[-offset:]
        marginal_count_probs = self._marginal_count_probs[:-offset] if offset>0  else self._marginal_count_probs[-offset:]
        tmp = logsumexp(log_prior[..., None]+count_log_probs[..., None, :], axis=1)
        return np.sum((tmp-marginal_count_probs[:, None, :])*np.exp(tmp), axis=(-2, -1))


class FullModel2(FullModel):
    def _init_calcs(self):
        # (v, gh, k)
        self._k_given_gh = np.array([[self._count_model.logpmf(k, self._N-k, G) for G in (0, 1, 2)]
                                     for k in range(self._N+1)]).T
        tmp = self._k_given_gh + self._genotype_log_probs[..., None]
        self._gh_given_k = tmp-logsumexp(tmp, axis=1, keepdims=True)

    def full_score_func(self, count_matrix, offset):
        count_matrix = count_matrix+ 0.1
        gm_given_gh = np.log(count_matrix/np.sum(count_matrix, axis=M, keepdims=True))
        gm_and_gh = np.log(count_matrix/np.sum(count_matrix, axis=(M, H), keepdims=True))
        gh_given_k = self._get_window(self._gh_given_k, offset)
        k_given_gh = self._get_window(self._k_given_gh, offset)
        p_hat = logsumexp(gh_given_k[..., None, :]+gm_given_gh[..., None], axis=1)
        p = logsumexp(k_given_gh[..., None, :]+gm_and_gh[..., None], axis=1)
        return np.sum(np.exp(p)*p_hat, axis=(-1, -2))
        
