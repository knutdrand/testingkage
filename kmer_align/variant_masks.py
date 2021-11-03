import numpy as np
from .helper_finder import BinomialModel, np_get_helper_to_predicted_helper_probs, get_helper_to_predicted_helper_probs
from .helper_variants import PriorModel

def get_masked_calc_func(score_func, mask):
    print("MASKED", np.sum(mask))
    mask = np.where(mask, -np.inf, 0)
    def masked_score_func(count_matrix, offset):
        m = mask[:-offset] if offset>0 else mask[-offset:]
        return score_func(count_matrix)+m
    return masked_score_func


def get_weighted_calc_func(score_func, weights, k=1):
    def weighted_score_func(count_matrix, offset):
        w = weights[:-offset] if offset>0 else weights[-offset:]
        return score_func(count_matrix)+w*k
    return weighted_score_func

def get_prob_weights(k_r, k_a, genotype_probs):
    model = BinomialModel(k_r, k_a)
    prior_model = PriorModel(model, np.log((genotype_probs)))
    prob_correct = get_prob_correct(prior_model)
    return np.log(prob_correct)

def get_prob_correct_mask(k_r, k_a, threshold,  genotype_probs):
    model = BinomialModel(k_r, k_a)
    t = np_get_helper_to_predicted_helper_probs(model, 20)
    prob_correct = np.sum(t.diagonal(axis1=-2, axis2=-1)*genotype_probs, axis=-1)
    return prob_correct < threshold


def get_prob_correct(model):
    N = 20
    correct_probs = np.zeros(model._n_variants)
    p_sum = np.zeros_like(correct_probs)
    for k in range(N+1):
        predicted = model.predict(k, N-k)
        for H in (0, 1, 2):
            p_k = np.exp(model.logpmf(k, N-k, H))
            p_sum += p_k
            # print(correct_probs.shape, (predicted==H).shape, p_k.shape)
            # print(correct_probs[predicted==H])
            correct_probs[predicted==H] += p_k[predicted==H]
    print(p_sum)
    print(correct_probs)
    return correct_probs

def get_prob_correct_mask_w_prior(k_r, k_a, threshold, genotype_probs):
    model = BinomialModel(k_r, k_a)
    prior_model = PriorModel(model, np.log((genotype_probs)))
    prob_correct = get_prob_correct(prior_model)
    # t = get_helper_to_predicted_helper_probs(prior_model, 20)
    # prob_correct = np.sum(t.diagonal(axis1=-2, axis2=-1)*genotype_probs, axis=-1)
    return prob_correct < threshold
