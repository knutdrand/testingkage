import numpy as np


def random_haplotype(n_variants, priors):
    assert sum(priors) == 1, priors
    return np.asanyarray(np.random.rand(n_variants) < priors[1], dtype="int")

def random_genotype(n_variants, priors):
    priors = np.asanyarray(priors)
    genotype = np.ones(n_variants, dtype="int")
    r = np.random.rand(n_variants)
    genotype[r<priors[0]] = 0
    genotype[r>1-priors[-1]] = 2
    return genotype
    
def simulate_genotype_matrix_from_founders(n_variants, n_individuals, founder_types, transition_prob=0.2):
    a = simulate_haplotype_matrix_from_founders(n_variants, n_individuals, founder_types, transition_prob)
    b = simulate_haplotype_matrix_from_founders(n_variants, n_individuals, founder_types, transition_prob)
    return a+b

def simulate_haplotype_matrix_from_founders(n_variants, n_individuals, founder_types, transition_prob=0.2):
    # founder_types = np.array([random_genotype(n_variants, priors) for _ in range(n_founders)]).T
    n_founders = founder_types.shape[-1]
    founder_changes = (np.random.rand(n_variants*n_individuals)<transition_prob).reshape(n_individuals, n_variants)
    founder_changes[:, 0] = 1
    founder_idxs = np.random.randint(0, n_founders, np.count_nonzero(founder_changes))
    idx_diffs = np.diff(founder_idxs)
    founders = np.zeros(n_variants*n_individuals,dtype="int")
    founders[np.flatnonzero(founder_changes.flatten())[1:]] = idx_diffs
    founders[0] = founder_idxs[0]
    founders = founders.cumsum().reshape(n_individuals, n_variants).T
    return founder_types[np.arange(n_variants)[:, None], founders]

# print(simulate_genotype_matrix_from_founders(3, 4, 2, transition_prob=0.3))

def simulate_genotype_matrix(n_variants, n_individuals, transition_prob=0.2):
    """ 
    Simulate genotype matrix
    return n_variants x n_individuals matrix
    """
    p = transition_prob
    q = 1-p
    transition_matrix = np.array([[q*q, 2*q*p, p**2],
                                  [q*p, p*p+q*q, p*q],
                                  [p**2, 2*q*p, q**2]])

    cum_transition_matrix = np.cumsum(transition_matrix, axis=1)
    matrix = []
    cur = np.random.choice([0, 1, 2], n_individuals)
    for v in range(n_variants):
        matrix.append(cur)
        rand = np.random.rand(n_individuals)
        new = np.ones_like(cur)
        new[rand<cum_transition_matrix[cur, 0]] = 0
        new[rand>cum_transition_matrix[cur, 1]] = 2
        cur = np.where(np.random.rand(n_individuals)<0.1, v%3,  new)

    return(np.array(matrix))


def simulate_kmer(p_duplicate=0.3, p_uncertain=0.5, duplicate_rate=2):
    if np.random.rand()<p_duplicate:
        n_duplicates = np.random.poisson(duplicate_rate)
        n_certain = np.random.binomial(n_duplicates, 1-p_uncertain)
        allele_frequencies = np.random.rand(n_duplicates-n_certain)
        return (n_certain, allele_frequencies)
    else:
        return (0, [])

def random_model(n_variants):
    ref_variants = [simulate_kmer(duplicate_rate=10, p_uncertain=0.2) for _ in range(n_variants)]
    alt_variants = [simulate_kmer(duplicate_rate=10, p_uncertain=0.2) for _ in range(n_variants)]
    return ref_variants, alt_variants

def get_instance(variants, n_copies):
    return np.array([certain*2+g+np.sum(np.random.rand(2*len(a))<np.tile(a, 2)) for (certain, a), g in zip(variants, n_copies)])

def simulate_counts(genotype, certain, probs, base_lambda=7.5):
    n_variants = genotype.shape[0]
    # genotype = genotype_matrix[:, np.random.randint(n_individs)]
    ref_variants = list(zip(certain[0], probs[0]))
    alt_variants = list(zip(certain[1], probs[1]))
    ref_instance = get_instance(ref_variants, 2-genotype)
    alt_instance = get_instance(alt_variants, genotype)
    ref_counts = np.random.poisson(ref_instance*base_lambda)
    alt_counts = np.random.poisson(alt_instance*base_lambda)
    return ref_counts, alt_counts
