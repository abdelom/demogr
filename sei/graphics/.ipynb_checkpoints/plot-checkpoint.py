"""
This module allows you to create graphics.
"""

import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np
from matplotlib.lines import Line2D


def normalization(data):
    """
    Data normalization to (0;1).
    """
    somme = sum(data)
    normalized_data = [ele / somme for ele in data]

    return normalized_data


def from_json_to_dataframe(path_data):
    """
    Export json file to pandas DataFrame.
    """
    dico = {
        "Parameters": np.array([], dtype=object), "Positive hit": np.array([], dtype=int),
        "Model0 ll": np.array([], dtype=list), "Model1 ll": np.array([], dtype=list),
        "SNPs": np.array([], dtype=int)
    }
    data = pd.DataFrame(dico)

    for fichier in [ele for ele in os.listdir(path_data)]:
        res = pd.read_json(path_or_buf="{}{}".format(path_data, fichier), typ='frame')
        # res['Tau'] = np.log10(res['Tau'])
        data = data.append(res, ignore_index=True)

    return data


######################################################################
# Plot Site Frequency Spectrum                                       #
######################################################################

def plot_sfs(sfs, label, color, title, style, axis=False, path_figure="./Figures/", name="sfs"):
    """
    Graphic representation of Site Frequency Spectrum (SFS), save to the folder ./Figures.

    Parameter
    ---------
    sfs: list
        list of sfs to plot
    label: list
        the label of each sfs
    color: list
        the color of each curve
    style: list
        the linestyle
    title: string
        title of the plot
    """
    # Plot
    for i, spectrum in enumerate(sfs):
        normalized_spectrum = normalization(spectrum)  # normalization of each sfs
        plt.plot(normalized_spectrum, color=color[i], linestyle=style[i], label=label[i])

    # Caption
    plt.legend(loc="upper right", fontsize="large")

    # Label axis
    plt.xlabel("Allele frequency", fontsize="large")
    plt.ylabel("Percent of SNPs", fontsize="large")

    # X axis values
    if axis:
        x_ax, x_values = [], []
        for i in range(len(sfs[0])):
            x_ax.append(i)
            x_values.append("{}/{}".format(i+1, len(sfs[0])+1))
        plt.xticks(x_ax, x_values)

    # Title + save plot to the folder ./Figures
    plt.title(title, fontsize="xx-large")
    plt.savefig("{}{}.png".format(path_figure, name))
    plt.clf()


######################################################################
# Plot for the optimization of grid size                             #
######################################################################

def plot_optimisation_grid(data, log_scale):
    """
    Plot for a given scenario the likelihood and optimal theta's value for various grid size.

    Parameter
    ---------
    data: dictionary
      - mu the rate of mutation
        - Likelihood
        - Estimated theta 
        - Theoritical theta

    log_scale:
    """
    fig, axs = plt.subplots(2, 2, sharex=True)

    cpt = 0
    mu, color = list(data.keys()), ['tab:red', 'tab:blue', 'tab:orange']

    for i in range(2):
        for j in range(2):
            ##############################################
            # Plots with different scale on the same fig #
            ##############################################

            # Set x_axis value
            x_ax = []
            for k in range(len(log_scale)):
                x_ax.append(k)
            axs[i, j].set_xticks(x_ax)  # value
            axs[i, j].set_xticklabels(log_scale)  # labels

            # Likelihood plot
            axs[i, j].plot(data[mu[cpt]]["Likelihood"], color=color[0])
            axs[i, j].tick_params(axis='y', labelcolor=color[0])

            # Theta plot
            ax2 = axs[i, j].twinx()  # instantiate a second axes that shares the same x-axis
            ax2.plot(data[mu[cpt]]["Estimated theta"], color=color[1])
            ax2.plot(data[mu[cpt]]["Theoritical theta"], color=color[2], linestyle='dashed')
            ax2.tick_params(axis='y', labelcolor=color[1])

            # Optimal value for grid size
            axs[i, j].axvline(x=4, color="tab:gray", linestyle="dashed")

            # Suptitle of each plot
            axs[i,j].set_title("Mutation rate {}".format(mu[cpt]))

            cpt +=1

    # Add common legend
    lg_ele = [
        Line2D([0], [0], linestyle='', marker='.', color=color[0], label='Likelihood'),
        Line2D([0], [0], linestyle='', marker='.', color=color[1], label='Estimated theta'),
        Line2D([0], [0], linestyle='', marker='.', color=color[2], label='Theoritical theta')
    ]
    legend = fig.legend(handles=lg_ele, loc='upper center', ncol=3, fontsize='medium',
                        bbox_to_anchor=(0., 1.05, 1., .102), borderaxespad=0.)
    fig.gca().add_artist(legend)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Add xlabel for bottom plot only
    cpt = 0
    for ax in axs.flat:
        if cpt > 1:
            ax.set(xlabel='Grid scale')
        cpt += 1

    # Title + save plot to the folder ./Figures
    fig.suptitle("Likelihood & theta's value for various grid point size", fontsize="x-large",
                 y=-0.05)
    plt.savefig("./Figures/optimisation_grid", bbox_inches="tight", dpi=300)
    plt.clf()


######################################################################
# Plot of error rate of dadi                                         #
######################################################################

def plot_error_rate(sample):
    """
    Plot the error rate of theta estimated for 100 inference with dadi.
    """
    # Read a csv file into a pandas DataFrame
    data = pd.read_csv("./Data/Error_rate/error-rate-{}.csv".format(sample), sep='\t')

    # Round value in execution time - some values not round for an unexpected reason
    tmp = pd.DataFrame(
        {"Execution time": [round(ele, 3) for ele in data['Execution time'].to_list()]}
    )
    data.update(tmp)

    # Plot
    sns.set_theme(style="whitegrid")
    ax = sns.boxplot(x="mu", y="Error rate", hue="Execution time", data=data,
                     width=0.45, dodge=False)

    # Set yaxis range
    ax.set(ylim=(0.85, 1.15))

    # Legend out of the plot
    ax.legend(loc='upper left', bbox_to_anchor=(1.04, 1), fontsize='small',
              borderaxespad=0., title="Average run time")

    # Title + save plot to folder ./Figures
    title = \
        """Error rate of 100 inferences with dadi for various mutation rate mu
        with n={} genomes sampled\n
        Each simulation is a constant model population
        """.format(sample)

    plt.title(title, fontsize="large", loc="center", wrap=True)
    plt.savefig("./Figures/Error_rate/error-rate-{}".format(sample), bbox_inches="tight")
    plt.clf()


######################################################################
# SNPs distribution for various tau                                  #
######################################################################

def snp_distribution():
    """
    Plot the SNPs distribution for various tau for kappa = 2 & kappa = 10.
    """
    path_data = "./Data/snp_distribution/"

    label = ['Kappa = 2', 'Fixed SNPs', 'Kappa = 10', 'Fixed SNPs']
    style = ['solid', 'dashed', 'solid', 'dashed']
    color = ['tab:blue', 'tab:blue', 'tab:orange', 'tab:orange']

    theoritical_theta = 32000

    _, axs = plt.subplots(1, 2, figsize=(15, 7))
    sns.set_theme(style="whitegrid")

    #######
    # Tau #
    #######
    for i, kappa in enumerate(['2', '2_fixed', '10', '10_fixed']):
        # Export data to DataFrame
        data = from_json_to_dataframe("{}Optimization_tau-kappa={}/".format(path_data, kappa))

        # Compute log10 of tau
        data['Tau'] = data['Parameters'].apply(lambda ele: np.log10(ele['Tau']))

        # Length factor
        if kappa == '2' or kappa == '10':
            length_factor = [round(np.mean(ele) / theoritical_theta, 1) for ele in data['SNPs']]
            with open("./Data/length_factor-kappa={}".format(kappa), 'w') as filout:
                for ele in sorted(length_factor, reverse=True):
                    filout.write("{} ".format(ele))

        # Compute mean of SNPs
        data['SNPs'] = data['SNPs'].apply(lambda ele: np.log10(np.mean(ele)))

        # Plot
        _ = sns.lineplot(
            x="Tau", y="SNPs", data=data, label=label[i], ax=axs[0], linestyle=style[i],
            color=color[i]
        )

    # Set axis label & title
    axs[0].set(xlabel='Log10(Tau)', ylabel='Log10(SNPs)',
               title='SNPs distribution for various tau - kappa fixed')

    #########
    # Kappa #
    #########
    # Export data to DataFrame
    data = from_json_to_dataframe("{}Optimization_kappa-tau={}/".format(path_data, 1))

    # Compute log10 of kappa
    data['Kappa'] = data['Parameters'].apply(lambda ele: np.log10(ele['Kappa']))

    # Length factor
    length_factor = [round(np.mean(ele) / theoritical_theta, 4) for ele in data['SNPs']]
    with open("./Data/length_factor-tau={}".format(1), 'w') as filout:
        for ele in sorted(length_factor):
            filout.write("{} ".format(ele))

    # Compute mean of SNPs
    data['SNPs'] = data['SNPs'].apply(lambda ele: np.log10(np.mean(ele)))

    # Plot
    _ = sns.lineplot(x="Kappa", y="SNPs", data=data, label='Tau = 1.0', ax=axs[1])

    # Set axis label & title
    axs[1].set(xlabel='Log10(Kappa)', ylabel='',
               title='SNPs distribution for various kappa - tau fixed')

    # Title + save plot to folder ./Figures
    plt.savefig("./Figures/snp_distribution", bbox_inches="tight")
    plt.clf()


######################################################################
# Plot SNPs distribution in 3D for various tau & kappa               #
######################################################################

def snp_distribution_3d(data):
    """
    Plot SNPs distribution in 3D of simulation with msprime.
    """
    # Compute log10 of kappa & tau
    data['Tau'] = data['Parameters'].apply(lambda ele: np.log10(ele['Tau']))
    data['Kappa'] = data['Parameters'].apply(lambda ele: np.log10(ele['Kappa']))

    # Compute mean of SNPs (then log10)
    data['SNPs'] = data['SNPs'].apply(lambda ele: np.log10(np.mean(ele)))

    # Plot
    sns.set_theme(style='whitegrid')
    sns.scatterplot(x='Tau', y='SNPs', size='Kappa', data=data, sizes=(10, 60))

    # Title + save plot to folder ./Figures
    plt.savefig("./Figures/snp_distribution_3d", bbox_inches="tight")
    plt.clf()


######################################################################
# Plot likelihood-ratio test                                         #
######################################################################

def plot_lrt(data, path="./Figures/"):
    """
    Plot the likelihood-ratio test.
    """
    # Compute log 10 of tau
    data['Tau'] = data['Parameters'].apply(lambda ele: np.log10(ele['Tau']))
    data['Positive hit'] = data['Positive hit'].astype(int)

    # Plot
    sns.set_theme(style="whitegrid")
    ax = sns.lineplot(x="Tau", y="Positive hit", data=data)

    # Set yaxis range
    ax.set(ylim=(0, max(data["Positive hit"]) + 5))

    # Title + save plot to folder ./Figures
    plt.title("Likelihood-ratio test - mu = 8e-2")
    plt.savefig("{}lrt".format(path), bbox_inches="tight")
    plt.clf()


######################################################################
# Plot weighted square distance d2                                  #
######################################################################

def compute_weighted_square_distance(sfs_a, sfs_b, case):
    """
    Compute the weighted square distance d2

    a) Between the observed SFS and the predicted SFS by model M1

    In this case d2 is:

      d2(eta_model, eta_observed) = Sum{(eta_model_i - eta_observed_i)^2 / eta_model_i}

    With
      - eta_observed the normalized observed SFS
      - eta_model the normalized inferred SFS of M1


    or b) Between the inferred SFS of two models (M0 & M1)

    In this case d2 is:

      d2(eta_m0, eta_m1) = Sum{(eta_m0_i - eta_m1_i)^2 / mean(eta_m0_i, eta_m1_i)}

    With
      - eta_m0 the normalized inferred SFS of M0
      - eta_m1 the normalized inferred SFS of M1
      - mean(eta_m0, eta_m1) = (eta_m0_i + eta_m1_i) / 2


    In both case, the sum is from i=1 to n (i=2 if singleton ignored)

    Parameter
    ---------
    case: either a or b
    """
    if case == 'a':
        d2_i = [
            np.power(eta_a - eta_b, 2) / eta_a for eta_a, eta_b in zip(sfs_a, sfs_b)
        ]
    else:
        d2_i = [
            np.power(eta_a - eta_b, 2) / np.mean([eta_a, eta_b]) for eta_a, eta_b
            in zip(sfs_a, sfs_b)
        ]
    return sum(d2_i)


def data_weighted_square_distance(data):
    """
    Retrieve and pre-process data that are needed for plot the weighted square distance d2.
    """
    # Normalized all SFS
    data['SFS obs'] = data['SFS obs'].apply(lambda x: [normalization(ele) for ele in x])
    data['SFS M0'] = data['SFS M0'].apply(lambda x: [normalization(ele) for ele in x])
    data['SFS M1'] = data['SFS M1'].apply(lambda x: [normalization(ele) for ele in x])

    # Compute weighted square distance for all SFS
    d2_model_inferred = {}  # d2 between the observed SFS and inferred SFS of M1
    d2_models = {}  # d2 between the inferred SFS of two models (M0 & M1)

    # Iterate over DataFrame row as (index, Series) pairs
    for i, row in data.iterrows():

        # Iterate over all the normalized observed SFS & normalized inferred SFS with M1
        tmp = []
        for sfs_a, sfs_b in zip(row['SFS M1'], row['SFS obs']):
            tmp.append(compute_weighted_square_distance(sfs_a, sfs_b, case='a'))
        d2_model_inferred[i] = tmp

        # Iterate over all the normalized observed SFS & normalized inferred SFS with M1
        tmp = []
        for sfs_a, sfs_b in zip(row['SFS M1'], row['SFS M0']):
            tmp.append(compute_weighted_square_distance(sfs_a, sfs_b, case='b'))
        d2_models[i] = tmp

    # Add new column to the DataFrame from dictionary - the dict keys need to be in the
    # DataFrame index
    data['d2 model inferred'] = pd.Series(d2_model_inferred)
    data['d2 models'] = pd.Series(d2_models)

    # Compute log 10 of tau
    data['Tau'] = data['Parameters'].apply(lambda ele: np.log10(ele['Tau']))

    return data


def plot_weighted_square_distance(data):
    """
    Plot the weighted square distance d2.
    """
    data = data_weighted_square_distance(data)

    # Plot the weighted square distance - median
    _, axs = plt.subplots(1, 2, figsize=(15, 7))
    sns.set_theme(style="whitegrid")

    percentiles, color, style = [2.5, 50, 97.5], ['tab:blue', 'tab:red'], ['dashed', 'solid', 'dashed']

    for i, centile in enumerate(percentiles):
        _ = sns.lineplot(
            x=data['Tau'], y=[np.percentile(ele, centile) for ele in data['d2 model inferred']],
            ax=axs[0], color=color[0], linestyle=style[i]
        )

        _ = sns.lineplot(
            x=data['Tau'], y=[np.percentile(ele, centile) for ele in data['d2 models']],
            ax=axs[1], color=color[1], linestyle=style[i]
        )

    # Title + save plot to folder ./Figures
    plt.savefig("./Figures/test", bbox_inches="tight")
    plt.clf()


if __name__ == "__main__":
    sys.exit()  # No actions desired
