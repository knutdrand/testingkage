"""Console script for kmer_align."""
import cProfile
import sys
import click
from .simulate import simulate_genotype_matrix, simulate_kmer, simulate_counts, random_genotype, random_haplotype, simulate_genotype_matrix_from_founders
from .poisson_model import ComboModel
from .both_alleles import ComboModelBothAlleles
from .helper_variants import HelperModel, PriorModel, SimpleHelperModel, calc_argmax, calc_likelihood
from .variant_masks import *
from .helper_finder import get_calc_func_with_model, get_calc_func
from .full_model import FullModel, FullModel2
from .evaluation import compare_models
from sklearn.metrics import confusion_matrix
import numpy as np

@click.command()
@click.option("-v", "--variants", default=100)
@click.option("-i", "--individs", default=100)
@click.option("-o", "--outfile", type=click.File("wb"))
def simulate(variants, individs, outfile):
    t_prob = 0.01
    founder_genotypes = np.array([random_haplotype(variants, [0.8, 0.2]) for _ in range(40)]).T
    M = simulate_genotype_matrix_from_founders(variants, individs, founder_genotypes, t_prob)
    # M = simulate_genotype_matrix(variants, individs, transition_prob=t_prob)
    alt_certain, alt_probs = zip(*[simulate_kmer(duplicate_rate=10, p_duplicate=0.5) for _ in range(variants)])
    ref_certain, ref_probs = zip(*[simulate_kmer(duplicate_rate=10, p_duplicate=0.5) for _ in range(variants)])
    certain = np.array([ref_certain, alt_certain])
    probs = np.array([ref_probs, alt_probs], dtype="object")
    # genotype = simulate_genotype_matrix(variants, 1, transition_prob=t_prob).flatten()
    genotype = simulate_genotype_matrix_from_founders(variants, 1, founder_genotypes, t_prob).flatten()
    counts = np.array(simulate_counts(genotype, certain, probs))
    np.savez(outfile, **{"M": M, "certain": certain, "probs": probs,
                         "genotype": genotype, "counts": counts})

@click.command()
@click.option("-i", "--infile", type=click.File("rb"))
@click.option("-o", "--outfile", type=click.File("wb"))
def predict(infile, outfile):
    simulated_data = np.load(infile, allow_pickle=True)
    certain_ref, certain_alt = simulated_data["certain"]
    probs_ref, probs_alt = simulated_data["probs"]
    ref_counts, alt_counts = simulated_data["counts"]
    ref_model = ComboModel.from_kmers(list(zip(certain_ref, probs_ref)), 7.5)
    alt_model = ComboModel.from_kmers(list(zip(certain_alt, probs_alt)), 7.5)
    model = ComboModelBothAlleles(ref_model, alt_model)
    #helper_model = HelperModel.from_genotype_matrix(model, simulated_data["M"])
    e_ref = certain_ref+np.array([sum(ps) for ps in probs_ref])
    e_alt = certain_alt+np.array([sum(ps) for ps in probs_alt])
    score_func = get_calc_func_with_model(e_ref, e_alt, 20)
    threshold=10
    mask = (e_ref>threshold) & (e_alt>threshold)
    masked_score_func = get_masked_calc_func(calc_likelihood, mask)
    predicted = model.predict(ref_counts, alt_counts)# , np.array([[0], [1], [2]], dtype="int"))
    # predicted = np.argmax(count_pmf, axis=1)
    genotype_counts = np.array([np.sum(simulated_data["M"]==i, axis=-1) for i in range(3)]).T
    mean_genotype_counts = np.mean(genotype_counts, axis=0)
    mean_genotype_counts /= np.sum(mean_genotype_counts)
    print("##################")
    print(mean_genotype_counts)
    genotype_counts = genotype_counts+mean_genotype_counts
    genotype_probs = genotype_counts/genotype_counts.sum(axis=-1, keepdims=True)
    genotype_matrix = simulated_data["M"]
    dummy_counts = mean_genotype_counts*mean_genotype_counts[:, None]
    models = {
        #"combo": model, 
        # "helper": HelperModel.from_genotype_matrix(model, simulated_data["M"], score_func=calc_likelihood),
        # "helper_w_model": HelperModel.from_genotype_matrix(model, simulated_data["M"], score_func=score_func, with_model=True),
        # "helper_w_naivmodel": HelperModel.from_genotype_matrix(model, simulated_data["M"], score_func=get_calc_func([np.eye(3) for _ in e_ref]), with_model=True),
        # "simple-helper": SimpleHelperModel.from_genotype_matrix(model, simulated_data["M"]),
        # "prior": PriorModel.from_genotype_matrix(model, simulated_data["M"])
    }
    full_score = FullModel(e_ref, e_alt, np.log(genotype_probs)).full_score_func
    # models["fullmodel"] = HelperModel.from_genotype_matrix(model, genotype_matrix, score_func=full_score, with_model=True, dummy_counts=dummy_counts)
    masks = [get_prob_correct_mask_w_prior(e_ref, e_alt, 0.35+0.03*i, genotype_probs) for i in range(1, 2)]
# for i, m in enumerate(masks):
#     models[f"probmask{0.35+0.03*i}"] = HelperModel.from_genotype_matrix(model, genotype_matrix, score_func=get_masked_calc_func(calc_likelihood, m), with_model=True, dummy_counts=dummy_counts)
# duplicate_masks = [(e_ref>t) & (e_alt>t) for t in range(6, 7)]
# for i, m in enumerate(duplicate_masks):
#     models[f"dupmask{i}"] = HelperModel.from_genotype_matrix(model, genotype_matrix, score_func=get_masked_calc_func(calc_likelihood, m), with_model=True, dummy_counts=dummy_counts)
    (m, f, g)  = (HelperModel, get_weighted_calc_func, get_prob_weights)
    cProfile.runctx("m.from_genotype_matrix(model, genotype_matrix, score_func=f(calc_likelihood, g(e_ref, e_alt, genotype_probs), 0.4), with_model=True, dummy_counts=dummy_counts, window_size=1000)", globals(), locals())
    for i in range(8, 9):
        models[f"weighted{0.05*i}"] = HelperModel.from_genotype_matrix(model, genotype_matrix, score_func=get_weighted_calc_func(calc_likelihood, get_prob_weights(e_ref, e_alt, genotype_probs), 0.05*i), with_model=True, dummy_counts=dummy_counts, window_size=1000)
    for name, m in models.items():
        print(name)
        c = confusion_matrix(simulated_data["genotype"], m.predict(ref_counts, alt_counts))
        print(c.diagonal()/c.sum(axis=-1)*100)
        print(c.trace()/c.sum()*100)
    # compare_models(models["helper"], models["fullmodel"], (ref_counts, alt_counts), simulated_data["genotype"])


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
