import msprime as ms
import numpy as np
import itertools as it
import time
import matplotlib.pyplot as plt
import pandas as pd
#import pickle as pkl
import seaborn as sns
import multiprocessing as mp
import math as mt
import random as rd

def msprime_simulate_variants(params, debug=False):
    """
    copyrigth pierre
    Population simulation with msprime for SMC++ (msprime 1.x).
    In this case, mutations are placed at discrete, integer coordinates => Msprime 1.01 is
    therefore needed (update of Msprime and tskit).
    Parameter
    ---------
    model: function
        (constant, sudden declin, sudden growth, etc.)
    params: dictionary
        - sample_size: the number of sampled monoploid genomes
        - Ne: the effective (diploid) population size
        - ro: the rate of recombinaison per base per generation
        - mu: the rate of infinite sites mutations per unit of sequence length per generation
        - length: the length of the simulated region in bases
    tau: the lenght of time ago at which the event (decline, growth) occured
    kappa: the growth or decline force
    debug: Boolean
        1: print msprime debugger, 0: nothing
    Some notes about the simulation of ancestry with the method sim_ancestry() of Msprime 1.x
      - samples
        The number of individual instead of the number of monoploid genomes (msprime 0.x)
      - ploidy
        Sets the default number of sample nodes (i.e. monoploid genomes) per individual
        Ploidy set to 2 means time to common ancestor in a population of size N is 2N
        generations (which is the same as msprime 0.x)
      - discrete_genome
        If True that means mutations are placed at discrete, integer coordinates
        If False that means mutations are placed at continuous, float coordinates (ms 0.x)
      If samples is set to 10 and ploidy to 1 there are N=10 genomes sampled
      If samples is set to 10 and ploidy to 2 there are 2N=20 genomes sampled
    Return
    ------
    sfs: list
        Site frequency Spectrum (sfs) - allele mutation frequency
    variants: list
        List of position and genotypes for each variant with 0 the ancestral state and 1 the
        alternative one.
    """
    # Set up the population model
    demography = ms.Demography()

    # Population actuelle au temps 0
    demography.add_population(initial_size=params['Ne'], growth_rate=0.)

    # Ancestral population
    demography.add_population_parameters_change(
        time=params['Tau'], population=0, initial_size=params['Ne']*params['Kappa'],
        growth_rate=0.)

    # if debug:
    #     print(demography.debug())

    # Simulation of ancestry
    ts = ms.sim_ancestry(
        samples=int(params['sample_size'] / 2), demography=demography, ploidy=2,
        sequence_length=params['length'], discrete_genome=True,
        recombination_rate=params['ro']
    )

    # Mutation model to use - binary mutation model
    #   - Allele ["0", "1"]
    #   - Root distribution [1., 0.], i.e. all ancestral states will be 0
    mutation_model = ms.BinaryMutationModel(state_independent=False)

    # Genetic variation of the data with mutation
    return ms.sim_mutations(tree_sequence=ts, rate=params['mu'], model=mutation_model)


# def nb_pair_distance(variants, params):
#     length = params["length"]
#     ld_distance = [0] * 100
#     d1 = int(np.floor(length * (1 - np.sqrt(1 - 1 / 99))))
#     f0 = (1 - d1 / length) ** 2
#     a = f0 / 99
#     variants, breakpoints = msprime_simulate_variants(params)
#     for variant1, variant2 in it.combinations(variants, 2):
#         distance = int(variant2.site.position - variant1.site.position)
#         ld_distance[int(np.floor((f0 - (1 - distance / length) ** 2) / a + 1))] += 1
#     return ld_distance


def linckage_desequilibrium(genotype_A_a, genotype_B_b):
    nb_sample = len(genotype_B_b)
    fA, fB = sum(genotype_A_a) / nb_sample, sum(genotype_B_b) / nb_sample
    fa, fb = 1 - fA, 1 - fB
    ld = sum(genotype_A_a & genotype_B_b) / nb_sample - fA * fB
    if ld < 0:
        a = - ld / min(fA * fB, fa * fb)
    else:
        a = ld / min(fA * fb, fa * fB)
    # print(a, genotype_A_a, genotype_B_b)
    return a


def length_mrf(breakpoints):
    breakpoints = list(breakpoints)
    return [breakpoints[i + 1] - breakpoints[i] for i in range(len(breakpoints) - 1)]
    # return [round(avg["total"] / avg["count"], 3) for avg in ld_bins]


def sfs(params):
    sfs = [0 for i in range(params["sample_size"])]
    variants = msprime_simulate_variants(params).variants()
    for variant in variants:
        sfs[sum(variant.genotypes)] += 1
    return np.array(sfs) / params["sample_size"]


def ld(params):
    ld_distance = [[0, 0, 0] for i in range(100)]
    variants = msprime_simulate_variants(params).variants()
    d1 = int(np.floor(params["length"] * (1 - np.sqrt(1 - 1 / 99))))
    f0 = (1 - d1 / params["length"]) ** 2
    a = f0 / 99
    list_snps = []
    variants = list(variants)
    p = 1400 / len(variants)
    for variant in variants:
        if len(set(variant.genotypes)) > 1 and rd.uniform(0, 1) < p:
            list_snps.append(variant)
    for variant1, variant2 in it.combinations(list_snps, 2):
        distance = int(variant2.site.position - variant1.site.position)
        ld = linckage_desequilibrium(variant1.genotypes, variant2.genotypes)
        index =  int(np.floor((f0 - (1 - distance / params["length"]) ** 2) / a + 1))
        ld_distance[index][0] += ld
        ld_distance[index][1] += distance
        ld_distance[index][2] += 1
    return [total / count for total, _, count in ld_distance]#, \
    # [total / count for _, total, count in ld_distance]


def replications(type, params, replicas):
    ld_cumul = np.zeros(100)
    for index in range(replicas):
        ld_cumul += np.array(type(params))
    ld_cumul = ld_cumul / replicas
    # parameters = {k: v for k, v in params.items() if k in ['Tau', 'Kappa']}
    # plot_ld((ld_cumul, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}),
    # "reference", True)
    return ld_cumul

# def scenario_replications(replicas):
#     for index in range(replicas):
#         yield msprime_simulate_variants(params).variants(), params["length"])
#     # parameters = {k: v for k, v in params.items() if k in ['Tau', 'Kappa']}
#     # plot_ld((ld_cumul, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}),
#     # "reference", True)


def chi2(type, params, kappa, tau):
    params.update({"Tau": tau, "Kappa": kappa})
    constant = params["constant"]
    chi2 = 0
    variation = replications(type, params, 20)
    for theoric, observed in  [*zip(constant, variation)]:
        chi2 += (observed - theoric) ** 2 / theoric
    return (np.log10(params["Tau"]), np.log10(params["Kappa"]),
                          np.log10(chi2))


def data_heat_map(type, kappa_range, tau_range, params):
    constant = replications(type, params, 200)
    params.update({"constant": constant})
    data = []
    pool = mp.Pool(mp.cpu_count())
    data = pool.starmap(chi2, [(type, params, kappa, tau) for kappa, tau in it.product(kappa_range, tau_range)])
    pool.close()
    return pd.DataFrame.from_records(data, columns = ['Tau', 'Kappa', 'Chi'])


def senario(type, params):
    d_kappa = {"Constant model": 1, "Modèle croissance": 0.1, "Modèle déclin": 10}
    for power in range(-2, -5, -1):
        data, parameters = {}, {}
        params.update({"ro": 8 * 10 ** power})
        for key, kappa in d_kappa.items():
            params.update({"Kappa": kappa, "Tau": 1})
            data[key] = replications(type, params, 100)
            parameters[key] = {k: v for k, v in params.items() if k in ['Tau', 'Kappa']}
    return data, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}
        # plot_ld((ld, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}),
        # "sce_ro/ld_int{}{}".format(power, parameter), True)
        # plot_ld((ld, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}),
        # "sce_ro/ld_distance{}{}".format(power, parameter), True,  distance)
        # plot_ld((distance, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}),
        # "sce_ro/distance_int{}{}".format(power, parameter), True)
        # boxplot_length_mrf(length_nrb,
        # "sce_ro/box_plot{}{}".format(power, parameter), True)
            # return ld, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}
#
#
# def LD_mu_senario(params, parameter="mu"):
#     d_mu = {"big": -3, "normal": -4, "small": -5}
#     # l_data = {}
#     for kappa in range(-1, 2, 1):
#         params.update({"Kappa": 10 ** kappa, "Tau": 1})
#         ld, length_nrb, parameters, distance = {}, {}, {}, {}
#         for key, power in d_mu.items():
#             params.update({parameter: 8 * 10 ** power})
#             variants, breakpoints = msprime_simulate_variants(params)
#             ld[key], distance[key] = all_ld(variants, params["length"])
#             length_nrb[key] = length_mrf(breakpoints)
#             parameters[key] = {k: v for k, v in params.items() if k in ['Tau', 'Kappa']}
#         plot_ld((ld, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}),
#         "mu_sce/ld_int{}{}".format(power, parameter), True)
#         plot_ld((ld, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}),
#         "mu_sce/ld_distance{}{}".format(power, parameter), True,  distance)
#         plot_ld((distance, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}),
#         "mu_sce/distance_int{}{}".format(power, parameter), True)
#         boxplot_length_mrf(length_nrb,
#         "mu_sce/box_plot{}{}".format(power, parameter), True)




if __name__ == "__main__":
    sys.exit()
