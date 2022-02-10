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


def nb_pair_distance(variants, params):
    length = params["length"]
    ld_distance = [0] * 100
    d1 = int(np.floor(length * (1 - np.sqrt(1 - 1 / 99))))
    f0 = (1 - d1 / length) ** 2
    a = f0 / 99
    variants, breakpoints = msprime_simulate_variants(params)
    for variant1, variant2 in it.combinations(variants, 2):
        distance = int(variant2.site.position - variant1.site.position)
        ld_distance[int(np.floor((f0 - (1 - distance / length) ** 2) / a + 1))] += 1
    return ld_distance


def all_ld(variants, length):
    ld_distance = [[0, 0, 0] for i in range(100)]
    d1 = int(np.floor(length * (1 - np.sqrt(1 - 1 / 99))))
    f0 = (1 - d1 / length) ** 2
    a = f0 / 99
    # print(len(variants))
    # print(len(variants) * (len(variants) - 1) / 2)
    for variant1, variant2 in it.combinations(variants, 2):
        distance = int(variant2.site.position - variant1.site.position)
        ld = linckage_desequilibrium(variant1.genotypes, variant2.genotypes)
        index =  int(np.floor((f0 - (1 - distance / length) ** 2) / a + 1))
        ld_distance[index][0] += ld
        ld_distance[index][1] += distance
        ld_distance[index][2] += 1
    return [total / count for total, _, count in ld_distance]#, \
    # [total / count for _, total, count in ld_distance]


# def all_ld(variants, sequence_length):
#     ld_distance = [0] * sequence_length
#     print(len(variants) * (len(variants) - 1) / 2)
#     for variant1, variant2 in tqdm(it.combinations(variants, 2)):
#         distance = int(variant2.site.position - variant1.site.position)
#         ld = linckage_desequilibrium(variant1.genotypes, variant2.genotypes)
#         if isinstance(ld_distance[distance], int):
#             ld_distance[distance] = [ld]
#         else:
#             ld_distance[distance].append(ld)
#     ld_distance = [elem for elem in ld_distance if isinstance(elem, int) == False]
#     distance = []
#     linckage = []
#     for d, sublist in enumerate(ld_distance):
#         distance += [d] * len(sublist)
#         for ld in sublist:
#             linckage.append(ld)
#     nb_elem = len(linckage) // 100
#     return [sum(linckage[i: i +  nb_elem]) / nb_elem for i in range(0, len(linckage) - nb_elem, nb_elem )], \
#     [sum(distance[i: i +  nb_elem]) / nb_elem for i in range(0, len(distance) - nb_elem, nb_elem )]


def length_mrf(breakpoints):
    breakpoints = list(breakpoints)
    return [breakpoints[i + 1] - breakpoints[i] for i in range(len(breakpoints) - 1)]
    # return [round(avg["total"] / avg["count"], 3) for avg in ld_bins]


# def nb_pair_distance(variants, nb_bins, sequence_length):
#     ld_bins = [0] * nb_bid
#     for variant1, variant2 in it.combinations(variants, 2):
#         distance = int(variant2.site.position - variant1.site.position)
#         ld_bins[distance // int(sequence_length / nb_bins)] += 1
#     return ld_bins


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
    ts = ms.sim_mutations(tree_sequence=ts, rate=params['mu'], model=mutation_model)
    # SVG(ts.draw_svg(y_axis=True))
    list_snps = []
    # return ts.variants()
    cmpt = 0
    p = 1400 / len(list(ts.variants()))
    for variant in ts.variants():
        if len(set(variant.genotypes)) > 1 and rd.uniform(0, 1) < p:
            cmpt += 1
            list_snps.append(variant)
    return list_snps #, ts.breakpoints()


def chi2(constant, params, kappa, tau):
    params.update({"Tau": tau, "Kappa": kappa})
    chi2 = 0
    variants = msprime_simulate_variants(params)
    variation = all_ld(variants, params["length"])
    for theoric, observed in  [*zip(constant, variation)]:
        chi2 += (observed - theoric) ** 2 / theoric
    return (np.log10(params["Tau"]), np.log10(params["Kappa"]),
                          np.log10(chi2) if chi2 > 0.01  else -2)

def LD_senario_ro(params, parameter="ro"):
    print("hello world !")
    d_kappa = {"Constant model": 1, "Modèle croissance": 0.1, "Modèle déclin": 10}
    for power in range(-2, -5, -1):
        ld, parameters, distance = {}, {}, {}
        params.update({parameter: 8 * 10 ** power})
        for key, kappa in d_kappa.items():
            params.update({"Kappa": kappa, "Tau": 1})
            start = time.time()
            print(start)
            variants, breakpoints = msprime_simulate_variants(params)
            print("time: {}".format(time.time() - start))
            ld[key], distance[key] = all_ld(variants, params["length"])
            # length_nrb[key] = length_mrf(breakpoints)
            parameters[key] = {k: v for k, v in params.items() if k in ['Tau', 'Kappa']}
        plot_ld((ld, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}),
        "sce_ro/ld_int{}{}".format(power, parameter), True)
        plot_ld((ld, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}),
        "sce_ro/ld_distance{}{}".format(power, parameter), True,  distance)
        plot_ld((distance, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}),
        "sce_ro/distance_int{}{}".format(power, parameter), True)
        # boxplot_length_mrf(length_nrb,
        # "sce_ro/box_plot{}{}".format(power, parameter), True)
    print("abdel")
            # return ld, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}


def LD_mu_senario(params, parameter="mu"):
    d_mu = {"big": -3, "normal": -4, "small": -5}
    # l_data = {}
    for kappa in range(-1, 2, 1):
        params.update({"Kappa": 10 ** kappa, "Tau": 1})
        ld, length_nrb, parameters, distance = {}, {}, {}, {}
        for key, power in d_mu.items():
            params.update({parameter: 8 * 10 ** power})
            variants, breakpoints = msprime_simulate_variants(params)
            ld[key], distance[key] = all_ld(variants, params["length"])
            length_nrb[key] = length_mrf(breakpoints)
            parameters[key] = {k: v for k, v in params.items() if k in ['Tau', 'Kappa']}
        plot_ld((ld, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}),
        "mu_sce/ld_int{}{}".format(power, parameter), True)
        plot_ld((ld, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}),
        "mu_sce/ld_distance{}{}".format(power, parameter), True,  distance)
        plot_ld((distance, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}),
        "mu_sce/distance_int{}{}".format(power, parameter), True)
        boxplot_length_mrf(length_nrb,
        "mu_sce/box_plot{}{}".format(power, parameter), True)

def contant_model(params, replicas):
    params.update({"Kappa": 1, "Tau": 1})
    ld_cumul, parameters = {"Constant model": np.zeros(100)}, {}
    for index in range(replicas):
        print(ld_cumul["Constant model"])
        ld_cumul["Constant model"] += np.array(all_ld(msprime_simulate_variants(params), params["length"]))
    parameters["Constant model"] = {k: v for k, v in params.items() if k in ['Tau', 'Kappa']}
    ld_cumul["Constant model"] = ld_cumul["Constant model"] / replicas
    # plot_ld((ld_cumul, parameters, {k: v for k, v in params.items() if k not in ['Tau', 'Kappa']}),
    # "reference", True)
    return ld_cumul["Constant model"]

def data_heat_map(kappa_range, tau_range, params):
    constant = contant_model(params, 100)
    data = []
    pool = mp.Pool(mp.cpu_count())
    data = pool.starmap(chi2, [(constant, params, kappa, tau) for kappa, tau in it.product(kappa_range, tau_range)])
    pool.close()
    return pd.DataFrame.from_records(data, columns =['Tau', 'Kappa', 'Chi'])
##############################################################################################################
##############################################Heatmap#########################################################
##############################################################################################################
def ld_label(length, title, save=False):
    """
    Set up sfs caption, label and title.

    Parameter
    ---------
    length: length of the SFS
    title: title of the plot
    """
    # Caption
    plt.legend(loc="upper right", fontsize="x-large")

    # Label axis
    if save:
        plt.xlabel("distance", fontsize="xx-large")
        plt.ylabel("linckage desequiibrium", fontsize="xx-large")
    else:
        plt.xlabel("distance", fontsize="x-large")
        plt.ylabel("linckage_desequilibrium", fontsize="x-large")
    # Title + show
    plt.title(title, fontsize="xx-large", fontweight='bold', y=1.01)


def boxplot_length_mrf(data, title_backup, save=False):
    fig = plt.figure(figsize=(12, 9))
    plt.boxplot(data.values())
    plt.xticks([1, 2, 3], data.keys())
    if save:
        plt.savefig('{}boxplot.png'.format(title_backup), format='png', dpi=150)

    plt.clf()


def plot_ld(data, title_backup, save=False, abs=None):
    """
    Graphic representation of Site Frequency Spectrum (SFS), save to the folder ./Figures.

    Parameter
    ---------
    data: tupple
        - 0: dictionary of {model: ld}
        - 1: dictionary of {model: model_parameter}, with  model_parameter either (tau, kappa)
          or (m12, kappa)
        - 2: dictionary of msprime simulation parameters

    save: boolean
        If set to True save the plot in ./Figures/ld-shape
    """
    color = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:gray"]
    labels = ['Modèle déclin', 'Modèle constant', 'Modèle croissance']

    # Set up plot
    plt.figure(figsize=(12, 9))  #, constrained_layout=True)
    cpt = 0
    for key, ld in data[0].items():
        # Label
        label = key
        for param, value in data[1][key].items():
            label += "{}={}".format(
                'τ' if param == 'Tau' else 'κ' if param == 'Kappa' else param, value
            )
            if param != list(data[1][key].keys())[-1]:
                label += ", "

        # Plot
        if abs is None:
            y = range(len(ld))
        else:
            y = abs[key]
        with plt.style.context('seaborn-whitegrid'):  # use seaborn style for plot
            plt.plot(y, ld, color=color[cpt], label=label,
                     marker='o' if key == 'Theoretical model' else '')
        cpt += 1

    # Label, caption and title
    if save:
        title = (
            "LD  pour différents scénarios avec Ne={}, mu={}, taux de recombinaison={}"
            " & L={:.1e}"
        ).format(data[2]['Ne'], data[2]['mu'], data[2]['ro'], data[2]['length'])
    else:
        title = (
            "LD with Ne={}, mutation rate mu={}, recombination rate={} & L={:.1e}"
        ).format(data[2]['Ne'], data[2]['mu'], data[2]['ro'], data[2]['length'])

    ld_label(length=len(ld), title=title, save=save)

    if save:
        plt.savefig('{}.png'.format(title_backup), format='png', dpi=150)
    plt.clf()


def heatmap_axis(ax, xaxis, yaxis, cbar, lrt=False):
    """
    Heatmap customization.

    Parameter
    ---------
    ax: matplotlib.axes.Axes
        ax to modify
    xaxis: str
        x-axis label
    yaxis: str
        y-axis label
    cbar: str
        colormap label
    """
    # Name
    names = [
        "$Log_{10}$" + "({})".format('τ' if xaxis == 'Tau' else xaxis),
        "$Log_{10}$" + "({})".format('κ' if yaxis == 'Kappa' else yaxis)
    ]  # (xaxis, yaxis)

    # x-axis
    plt.xticks(
        np.arange(64, step=7) + 0.5,
        labels=[round(ele, 2) for ele in np.arange(-4, 2.5, 0.7)],
        rotation='horizontal'
    )

   # plt.xticks(
    #    np.arange(3) + 0.5,
     #   labels=[round(ele, 2) for ele in np.arange(-1.0, 2.0, 1.0)],
      #  rotation='horizontal'
    #)
    plt.xlabel(names[0], fontsize="large")

    #y-axis
   # ax.set_ylim(ax.get_ylim()[::-1])  # reverse y-axis
    plt.yticks(
      np.arange(64, step=7) + 0.5,
      labels=[round(ele, 2) for ele in np.arange(-3.5, 3, 0.7)]
    )
    #plt.yticks(
     #   np.arange(3) + 0.5,
      #  labels=[round(ele, 2) for ele in np.arange(-1.0, 2.0, 1.0)]
    #)
    plt.ylabel(names[1], fontsize="large")

    # Set colorbar label & font size
    ax.figure.axes[-1].set_ylabel(cbar, fontsize="large")
    if lrt:
        ax.collections[0].colorbar.set_ticks([0, 20., 40., 60., 80., 100.])
        ax.collections[0].colorbar.set_ticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])

    # Set axis label of kappa = 0.0, i.e. constant model
    for i, ele in enumerate(plt.gca().get_yticklabels()):
        print(i, ele)
    index = [
        i for i, ele in enumerate(plt.gca().get_yticklabels()) if ele.get_text() == "0.0"
    ][0]  # get the index of kappa = 0.0
    plt.gca().get_yticklabels()[index].set_color('#8b1538')  # set the color
    plt.gca().get_yticklabels()[index].set_fontsize('medium')  # set the size
    plt.gca().get_yticklabels()[index].set_fontweight('bold')  # set the weight

    # Hlines for kappa = 0
    ax.hlines([35, 36], *ax.get_xlim(), colors="white", lw=2.)
    ax.vlines([0, 65], ymin=35, ymax=36, color="white", lw=2.)


def plot_heatmap(data, title, cbar, filout="heatmap.png", lrt=False):
    """
    Heatmap

    Parameter
    ---------
    data: pandas DataFrame
    """
    # Set up plot
    plt.figure(figsize=(12, 9), constrained_layout=True)
    sns.set_theme(style='whitegrid')

    # Pre-processing data
    df = data.pivot(index=data.columns[1], columns=data.columns[0], values=data.columns[2])
    print(df)
    # Plot
    ax = sns.heatmap(df, cmap='viridis')

    # Heatmap x and y-axis personnalization
    heatmap_axis(ax=ax, xaxis=df.columns.name, yaxis=df.index.name, cbar=cbar, lrt=lrt)

    # Title
    plt.title(title, fontsize="large", fontweight='bold', pad=10.5)

    plt.savefig(filout, format='png', dpi=150)
    plt.plot()


def main():
    params = {"sample_size":10, "Ne": 1, "ro": 8e-3, "mu": 8e-3,  "Tau": 1.0, "length": int(1e5)}
    kappa_range = np.exp(np.arange(-4, 2.8, 0.1))
    tau_range= np.exp(np.arange(-3.5, 2.3, 0.1))
    data  = data_heat_map(kappa_range, tau_range, params)
    data.to_csv("./data_ht", index=False)
    # plot_heatmap(data=data, title=title, cbar=cbar, filout='heatmap_test.png')
       # pkl.dump(generat_senar(params), "out")
     # params = {"sample_size":10, "Ne": 1, "ro": 8e-2, "mu": 8e-3,  "Tau": 1.0, "length": int(1e5)}
     # LD_senario_ro(params, "ro")
     # LD_mu_senario(params, "mu")

     # print(data)


if __name__ == "__main__":
    main()
