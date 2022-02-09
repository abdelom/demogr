"""
This module allows you to read or write files.
"""

import ast
import copy
import csv
import json
import os
import sys
import numpy as np
import pandas as pd

from itertools import islice


def zip_file(data):
    """
    Method to zip a file.
    """
    os.system("zip -rj {0}.zip {0}".format(data))
    os.system("rm -rf {}".format(data))


######################################################################
# SFS shape verification                                             #
######################################################################

def load_sfs(function, generate=False):
    """
    If generate True, create a new set of SFS for various scenario.

    If generate False, load a set of SFS for various scenario.
    """
    if generate:
        all_sfs, params, params_simulation = function()
        with open("./Data/Msprime/sfs_shape_verification", 'w') as filout:
            filout.write("SFS shape verification and simulations parameters - {}\n"
                         .format(params_simulation))

            for model, sfs in all_sfs.items():
                if model in params.keys():
                    filout.write("{} - {} - {}\n".format(model, sfs, params[model]))
                else:
                    filout.write("{} - {}\n".format(model, sfs))

    else:
        all_sfs, params = {}, {}
        with open("./Data/Msprime/sfs_shape_verification", 'r') as filin:
            lines = filin.readlines()
            params_simulation = ast.literal_eval(lines[0].strip().split(' - ')[1])

            for line in lines[1:]:
                tmp = line.strip().split(' - ')
                if tmp[0] not in ['Constant model', 'Theoretical model']:
                    params[tmp[0]] = ast.literal_eval(tmp[2])
                all_sfs[tmp[0]] = json.loads(tmp[1])

    return all_sfs, params, params_simulation


######################################################################
# SFS - species (real data)                                          #
######################################################################

def load_species_sfs(species):
    """
    Load the SFS of a given species (real data).

    Parameter
    ---------
    species: str
        A given species

    Return
    ------
    SFS: list
        The SFS for a gicen species
    """
    sfs = []
    with open("./Data/Real_data/SFS/{}".format(species.replace(' ', '_')), 'r') as filin:
        lines = filin.readlines()
        for line in lines:
            sfs.append(int(line.strip().split('\t')[1]))
    return sfs


def load_species_data():
    """
    Load for each species (real data) the SFS.

    Return
    ------
    data: dictionary
        Dictionary of species (key) and SFS (value)
    """
    data = {}
    with open("./Data/Real_data/Species.csv", "r") as filin:
        reader = csv.DictReader(filin)
        for row in reader:
            data[row['Species']] = {
                'SFS': load_species_sfs(row['Species']), 'Status': row['Conservation status']
            }
    return data


######################################################################
# SFS - Dadi                                                         #
######################################################################

def dadi_data(sfs_observed, fichier, fold, path="./Data/", name="SFS"):
    """
    Create SFS of a scenario in the format compatible with the dadi software.

    (Gutenkunst et al. 2009, see their manual for details)

    A pre-processing of the SFS is needed
      1. Which one
         Adding 0/n & n/n.
      2. Why
         Spectrum arrays are masked, i.e. certain entries can be set to be ignored
         For example, the two corners [0,0] corresponding to variants observed in zero samples
         or in all samples are ignored

    Parameter
    ---------
    sfs_observed: list
        the original SFS - without corners 0/n & n/n
    fichier: str
        file in which the SFS will be written in the format compatible with dadi
    fold: bool
        if the SFS must be fold (True) or not (False)
    """
    # Pre-processing of the SFS
    # sfs = copy.deepcopy(sfs_observed)
    sfs = [0] + sfs_observed + [0]  # Add 0/n & n/n to the sfs (lower and upper bound)
    if fold:
        sfs = sfs[:round(len(sfs)/2) + 1] + [0] * int(np.floor(len(sfs)/2))  # SFS folded

    with open("{}{}.fs".format(path, name), "w") as filout:
        if fold:
            filout.write("{} folded \"{}\"\n".format(len(sfs), fichier))
        else:
            filout.write("{} unfolded \"{}\"\n".format(len(sfs), fichier))


        # Write the SFS
        for freq in sfs:
            filout.write("{} ".format(freq))

        # Write 1 to mask value & 0 to unmask value
        filout.write("\n")
        for freq in sfs:
            filout.write("1 ") if freq == 0 else filout.write("0 ")

    del sfs


######################################################################
# SFS - Stairway plot 2                                              #
######################################################################

def stairway_data(name, data, path, fold):
    """
    Create SFS of a scenario in the format compatible with the stairway plot v2 software.

    (Xiaoming Liu & Yun-Xin Fu 2020, stairway-plot-v2, see readme file for details)
    """
    sfs, nseq, mu, year, ninput = \
        data['sfs'], data['sample_size'], data['mu'], data['year'], data['ninput']
    length = sum(sfs) * 10

    sfs = sfs[:round(len(sfs)/2)] if fold else sfs

    with open("{}{}.blueprint".format(path, name), "w") as filout:
        filout.write("# Blueprint {} file\n".format(name))
        filout.write("popid: {} # id of the population (no white space)\n".format(name))
        filout.write("nseq: {} # number of sequences\n".format(nseq))
        filout.write(
            "L: {} # total number of observed nucleic sites, including poly-/mono-morphic\n"
            .format(length))
        filout.write("whether_folded: {} # folded (true) or unfolded (false) SFS\n"
                     .format(fold))

        # SFS
        filout.write("SFS: ")
        for snp in sfs:
            filout.write("{} ".format(snp))
        filout.write("# snp frequency spectrum: number of singleton, doubleton, etc.\n")

        filout.write("#smallest_size_of_SFS_bin_used_for_estimation: 1\n")
        filout.write("#largest_size_of_SFS_bin_used_for_estimation: {}\n"
                     .format(round(nseq/2) if fold else nseq-1))
        filout.write("pct_training: 0.67 # percentage of sites for training\n")

        # Break points
        break_points = np.ceil([(nseq-2)/4, (nseq-2)/2, (nseq-2)*3/4, (nseq-2)])
        filout.write("nrand: ")
        for ele in break_points:
            filout.write("{} ".format(int(ele)))
        filout.write("# number of random break points for each try\n")

        filout.write("project_dir: {}{} # project directory\n".format(path, name))
        filout.write("stairway_plot_dir: {}stairway_plot_es\n".format(path))
        filout.write("ninput: {} # number of input files to be created for each estimation\n"
                     .format(ninput))
        filout.write("# random_seed: 6  if commented, the program will randomly pick one\n")

        # Output
        filout.write("# Output settings\n")
        filout.write("mu: {} # assumed mutation rate per site per generation\n".format(mu))
        filout.write(
            "year_per_generation: {} # assumed generation time (in year)\n".format(1))

        # Plot
        filout.write("# Plot settings\n")
        filout.write("plot_title: {} # title of the plot\n".format(name))
        filout.write(
            "xrange: 0,0 # Time (1K year) range; format: xmin, xmax; 0,0 for default\n")
        filout.write(
            "yrange: 0,0 # Ne (1k individual) range; format: ymin, ymax; 0,0 for default\n")
        filout.write("xspacing: 2 # x axis spacing\n")
        filout.write("yspacing: 2 # y axis sapcing\n")
        filout.write("fontsize: 12\n")


def read_stairway_final(path):
    """
    Read all file from the final folder, folder generated by stairway at the end of the
    inference.

    Parameter
    ---------
    path: path of the final folder

    Return
    ------
    data: dictionary
      - M0: default model, i.e. model of 1 dimension
        with LL: the log-likelihood of the testing data - stairway return a minus log-likelihood
             Theta: theta of of M0
      - M1: the final model with x dimension
        with LL: idem
             Theta min: minimum theta
             Theta max: maximum theta
    """
    data = {'M0': {'LL': [], 'Theta': []}, 'M1': {'LL': [], 'Theta': []},
            'Final': {'LL': [], 'Theta': []}}

    for fichier in os.listdir(path):
        with open("{}{}".format(path, fichier), 'r') as filin:
            lines = filin.readlines()

            # Read lines that start with dim
            models = [line.strip().split('\t') for line in lines[:-2] if line.startswith('dim')]

            # M0: Log likelihood and theta for 1 dimensional model, i.e. the constant model
            _, _, _, ll, _, _, _, theta = models[0]
            data['M0']['LL'].append(-float(ll))
            data['M0']['Theta'].append(float(theta))

            # M1: Log likelihood and theta for 2 dimensionals model, i.e. the various pop model
            if len(models) == 1:
                for value in data['M1'].values():
                    value.append(np.nan)
            else:
                _, _, _, ll, _, _, _, _, theta1, theta2 = models[1]
                theta = [float(theta1), float(theta2)]
                data['M1']['LL'].append(-float(ll))
                data['M1']['Theta'].append((min(theta), max(theta)))  # pair Theta min & max

            # Final: Log likelihood and theta for the final model - x dimension
            _, ll, _ = lines[-3].strip().split('\t')  # line that start with final
            data['Final']['LL'].append(-float(ll))

            line = [float(ele) for ele in lines[-1].strip().split(' ')]
            #data['Final']['Theta'].append((min(line), max(line), len(line)))
            data['Final']['Theta'].append(line)

    return data


def min_max(data):
    """
    Compute the min & max (value and index) of a list

    Return
    ------
    Pair Ne min (value, [index]) & Ne max (value, [index])
    """
    minimum, maximum = min(data), max(data)
    index_min = [i for i, ele in enumerate(data) if ele == minimum]
    index_max = [i for i, ele in enumerate(data) if ele == maximum]
    return (minimum, index_min), (maximum, index_max)


def read_stairway_summary(fichier):
    """
    Read the final output summary of the inference with stairway.

    Parameter
    ---------
    fichier: file to read

    Return
    ------
    data: dictionary
      - Ne: pair (Ne min, Ne max)
      - Ne initial: Ne of the initial population, i.e. at time 0
      - Ne ancestral: the oldest Ne
      - Ne mean: mean of Ne
      - Year: pair (Year of Ne min, Year of Ne max)
    """
    with open(fichier, "r") as filin:
        lines = filin.readlines()
        ne_list, year_list = [], []

        for line in lines[1:]:
            _, _, _, _, _, year, ne, _, _, _, _ = line.strip().split('\t')
            ne_list.append(float(ne))
            year_list.append(float(year))

        # Pair Ne min (value, [index]) & Ne max (value, [index])
        minimum, maximum = min_max(ne_list)

    data = {
        'Ne': (minimum[0], maximum[0]),
        'Ne initial': ne_list[0], 'Ne ancestral': ne_list[-1], 'Ne mean': np.mean(ne_list),
        'Year': ([year_list[i] for i in minimum[1]], [year_list[i] for i in maximum[1]])
    }

    return data


######################################################################
# Export json files                                                  #
######################################################################

def extract_param(fichier):
    """
    Extract from a file name the parameters and values.
    """
    fichier = fichier[:-4]
    return {
        ele.split('=')[0]: float(ele.split('=')[1]) for ele in fichier.rsplit('_', 2)[1:]
    }


def export_simulation_files(typ, job, path_data, param=None, value=None):
    """
    Export each json file generated with msprime into a single DataFrame.
    Then export this DataFrame to a json file.

    Parameters
    ----------
    typ: either SFS or VCF
    job
    param
    value
    """
    fichiers = [filin for filin in os.listdir(path_data) if filin.startswith(typ.upper())]

    # Selection of the file
    if param is not None:
        fichiers = [
            fil for fil in os.listdir(path_data) if extract_param(fil)[param] == value
        ]

    # Load Data
    simulation = pd.read_json("{}{}".format(path_data, fichiers[job]))
    
    if typ == 'SFS':
        return simulation[['Parameters', 'SFS observed', 'SNPs', 'Time']].iloc[0]

    return simulation.iloc[0]


def export_inference_files(model, fold, param, value=None):
    """
    Export each json file generated with dadi into a single DataFrame.
    Then export this DataFrame to a json file.

    Parameter
    ---------
    model: either cst, decline or migration
    param: either all, tau, kappa or ne
    value: not None if param is tau or kappa, it's the value of the fixed parameters for the
    inference
    """
    # Data
    col = ['Parameters', 'Positive hit', 'SNPs', 'SFS observed', 'M0', 'M1', 'Time',
           'd2 observed inferred', 'd2 models']
    inference = pd.DataFrame(columns=col)
    inference['Positive hit'] = inference['Positive hit'].astype(int)

    # Path data and filin
    path_data = "./Data/Dadi/{}/{}/".format(model, param)
    path_data += "Folded/" if fold else "Unfolded/"
    if param == 'all':
        filin = "dadi_{}_all".format(model)
    else:
        filin = "dadi_{}={}_all".format(model, value)

    # Read file
    if "{}.zip".format(filin) not in os.listdir(path_data):

        fichiers = os.listdir(path_data)

        # Select estimation for the specific value of param that is either tau, kappa or m12
        if param != 'all':
            fichiers = [
                fichier for fichier in fichiers
                if not fichier.endswith('all.zip')
                and float(fichier.rsplit('-', maxsplit=1)[0].split('=')[1]) == value
            ]

        for fichier in fichiers:
            res = pd.read_json(path_or_buf="{}{}".format(path_data, fichier), typ='frame')
            inference = inference.append(res, ignore_index=True)

            # Delete the json file
            os.remove("{}{}".format(path_data, fichier))

        # Export pandas DataFrame inference to json file
        inference.to_json("{}{}".format(path_data, filin))

    else:
        inference = pd.read_json(path_or_buf="{}{}.zip".format(path_data, filin), typ='frame')

    # Zip file
    zip_file("{}{}".format(path_data, filin))

    return inference


def export_specific_dadi_inference(model, fixed_param, values, fold):
    """
    Export specific dadi inference file for a given fixed parameter.

    A detailed application of this method can be found in the notebook analyse.ipynb.

    Parameter
    ---------
    model: str
        either decline or migration
    fixed_param: str
        fixed parameter to consider - either kappa, tau or m12
    values: list of float
        fixed parameter value to consider - log scale
        - Tau/m12: between -4 included and 2.5 excluded
        - Kappa: between -3.5 included and 3 excluded
    fold: bool
        True : inference is done with folded SFS
        False: inference is done with unfolded SFS

    Return
    ------
    data: list of pandas DataFrame of dadi inference
    labels: list of label for each DataFrame
    """
    data, labels = [], []

    for val in values:
        data.append(export_inference_files(model, fold, fixed_param, val))
        labels.append("{} = {:.1e}".format(
            fixed_param if fixed_param == "m12" else fixed_param.capitalize(),
            np.power(10, val))
        )

    return data, labels


# Stairway files

def export_stairway_files(model, fold):
    """
    Export each file generated from the inference with stairway into a single DataFraùe.
    Then export this DataFrame to a json file.

    Parameter
    ---------
    model: either decline or migration
    fold: boolean
      - True: inference with folded SFS
      - False: inference with unfolded SFS
    """
    # Path data and filin
    path_data = "./Data/Stairway/{}/".format(model)
    path_data += "Folded/" if fold else "Unfolded/"

    filin = "stairway_inference_{}".format(model)

    if "{}.zip".format(filin) not in os.listdir(path_data):
        # Data
        inference = pd.DataFrame()

        # Read all files
        fichiers = os.listdir(path_data)
        for fil in fichiers:
            # Read the file
            res = pd.read_json("{}{}".format(path_data, fil), typ='frame')
            inference = inference.append(res, ignore_index=True)

            # Delete the file
            os.remove("{}{}".format(path_data, fil))

        # Export pandas DataFrame to json file
        inference.to_json("{}{}".format(path_data, filin))

        # Zip
        os.system('zip -j {0}{1}.zip {0}{1}'.format(path_data, filin))
        os.remove("{}{}".format(path_data, filin))

    else:
        inference = pd.read_json("{}{}.zip".format(path_data, filin), typ='frame')

    return inference


######################################################################
# SMC++ file                                                        #
######################################################################

def variants_to_vcf(variants, param, fichier, path_data, ploidy=2):
    """
    Writes a VCF formatted file from variants generated with msprime.

    Parameter
    ---------
    variants: list
        List of position and genotype for each variant with 0 the ancestral state and 1 the
        alternative one.
    param:
      - sample: sample size
      - length: the length L of the sequence
    ploidy: int
        The ploidy of the individual samples
        By default it's set to 2, so for a sample of size 20 we will have 10 diploid samples in
        the output, consisting of the combined allele of [0, 1], [2, 3], ..., [18, 19].
    """
    if param['sample_size'] % ploidy != 0:
        sys.exit("Error \"variants_to_vcf\": sample size must be divisible by ploidy")

    with open("{}{}".format(path_data, fichier), 'w') as filout:
        # Write the header
        filout.write("##fileformat=VCFv4.2\n")
        filout.write("##source=tskit 0.3.4\n")
        filout.write("##FILTER=<ID=PASS,Description=\"All filters passed\">\n")
        filout.write("##contig=<ID=1,length={}>\n".format(param['length']))
        filout.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")

        # Write the genotype
        header = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO', 'FORMAT']
        header += ['tsk_{}'.format(i) for i in range(round(param['sample_size'] / ploidy))]
        filout.write("\t".join(header) + "\n")

        # For SMC++, discrete_genome parameter of the method sim_ancestry() of Msprime is set to
        # True, i.e. mutations are placed at discrete, integer coordinates
        # For Msprime, the coordinate's range is [0, L-1] with L the size of the sequence
        #   For SMC++, the coordinate's range is [1, L]
        # So each position is increased by 1
        for variant in variants:
            value = [
                '1', str(variant[0] + 1), '.', '0', '1', '.', 'PASS', '.', 'GT'
            ]
            if ploidy == 1:
                value += [str(genotype) for genotype in variant[1]]
            else:
                value += [
                    "{}|{}".format(variant[1][i], variant[1][i+1]) for
                    i in range(0, param['sample_size'], ploidy)
                ]
            filout.write("\t".join(value) + "\n")


def vcf_to_smc(fichier, path_data):
    """
    Convert a VCF file to SMC++ file.
    """
    # Get the fifth line of the VCF (doesn't read all the file)
    with open("{}{}".format(path_data, fichier), "r") as filin:
        line = next(islice(filin, 5, 6)).strip().split('\t')[9:]

    # Zip the VCF file with bgzip
    os.system("bgzip -f {}{}".format(path_data, fichier))

    # Generate index (CSI format) for bgzip compressed VCF files
    os.system("bcftools index -cf {}{}.gz".format(path_data, fichier))

    # VCF file to SMC++ format - generate the command
    command = (
        "smc++ vcf2smc {0}{1}.gz {0}smc_{2}.gz {contig} "
    ).format(path_data, fichier, fichier.split('_', 1)[1], contig=1)

    command += "Pop1:"
    for member in line:
        command += "{}".format(member) if member == line[-1] else "{},".format(member)

    # VCF file to SMC++ format - execute the command
    os.system(command)


if __name__ == "__main__":
    sys.exit()  # No actions desired
