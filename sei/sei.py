"""
Decline estimation from genomic data.
"""

import copy
import os
import sys
import time
import warnings
import pandas as pd
import numpy as np
from scipy.stats import chi2

import sei.arguments.arguments as arg
import sei.files.files as f
import sei.graphics.plot as plot
import sei.inference.dadi as dadi
import sei.simulation.msprime as ms


def computation_theoretical_theta(ne, mu, length):
    """
    Compute the theoretical theta - theta = 4.Ne.mu.L with

      - 4: diploid population
      - Ne: the effective population size
      - mu: the mutation rate
      - L: the size of the simulated genomes
    """
    return 4 * ne * mu * length


def simulation_parameters(sample, ne, rcb_rate, mu, length):
    """
    Set up the parametres for the simulation with msprime.
    """
    parameters = {
        "sample_size": sample, "Ne": ne, "rcb_rate": rcb_rate, "mu": mu, "length": length
    }
    return parameters


def define_parameters(model, typ):
    """
    Define pairs of (Tau, Kappa) - sudden decline/growth model - and (m12, Kappa) - migration
    model.
    """
    tau_range = np.arange(-4, 2.5, 0.1) if typ == 'sfs' else np.arange(-2, 1.1, 0.1)
    kappa_range = np.arange(-3.5, 3, 0.1) if typ == 'sfs' else np.arange(-1.5, 1.6, 0.1)

    if model == 'decline':
        # Range of value for tau & kappa
        params = []
        for tau in tau_range:
            for kappa in kappa_range:
                params.append({'Tau': round(tau, 2), 'Kappa': round(kappa, 2)})

    else:
        # Range of value for m12 & kappa
        params = []
        for m12 in tau_range:
            for kappa in kappa_range:
                params.append({'m12': round(m12, 2), 'm21': 0.0, 'Kappa': round(kappa, 2)})

    return params


######################################################################
# SFS shape verification                                             #
######################################################################

def compute_theoretical_sfs(length):
    """
    Compute the theoretical SFS of any constant population.
    """
    return [1 / (i+1) for i in range(length)]


def generate_set_sfs():
    """
    Generate a set of sfs for various scenario with msprime:

      - Constant population
      - Theoretical SFS for any constant population
      - Sudden growth model - growth of force kappa at a time tau in the past
      - Sudden decline model - decline of force kappa at a time tau in the past
      - Migration model - migration into population 1 from 2 and no migration into 2 from 1

    Return
    ------
    sfs: dictionary
        sfs for various scenario
    parameters: dictionary
        specific value of tau & kappa for the growth and decline model
        specifiv value of m12, m21 & kappa for the migration model
    params_simulation:
        Ne, sample size, length, mutation & recombination rate
    """
    sfs, parameters = {}, {}

    # Parameters for the simulation
    params = simulation_parameters(sample=10, ne=1, rcb_rate=2e-2, mu=2e-2, length=1e5)

    # Constant scenario
    sfs['Constant model'], _ = \
        ms.msprime_simulation(model=ms.constant_model, params=params, debug=True)

    # Theoretical SFS for any constant population
    sfs['Theoretical model'], _ = \
        compute_theoretical_sfs(length=params["sample_size"] - 1)

    # Define tau & kappa for decline scenario
    params.update({"Tau": 1., "Kappa": 10.})

    sfs['Decline model'], _ = \
        ms.msprime_simulation(model=ms.sudden_decline_model, params=params, debug=True)
    parameters['Decline model'] = {k: v for k, v in params.items() if k in ['Tau', 'Kappa']}

    # Define tau & kappa fo, _r growth scenario
    params.update({"Tau": 1., "Kappa": 0.1})

    sfs['Growth model'], _ = \
        ms.msprime_simulation(model=ms.sudden_decline_model, params=params, debug=True)
    parameters['Growth model'] = {k: v for k, v in params.items() if k in ['Tau', 'Kappa']}

    # Migration scenario, _
    # params.update({"Kappa": 10., "m12": 1., "m21": 0})

    # sfs['Migration model'], _ = \
    #     ms.msprime_simulation(model=ms.twopops_migration_model, params=params, debug=True)
    # parameters['Migration model'] = {
    #     k: v for k, v in params.items() if k in ['m12', 'm21', 'Kappa']
    # }, _

    params_simulation = \
        {k: v for k, v in params.items() if k not in ['m12', 'm21', 'Kappa', 'Tau']}

    return sfs, parameters, params_simulation


######################################################################
# Generate a set of Data with msprime                                #
######################################################################

def length_from_file(fichier, params, mu, snp):
    """
    Extract length factor from file and return the length of the sequence.
    """
    res = pd.read_json(path_or_buf="{}".format(fichier), typ='frame')

    # Bug of to_json or from_json method of pandas
    # Some value of tau, kappa or m12 with many decimal points
    for i, row in res.iterrows():
        res.at[i, 'Parameters'] = {k: round(v, 2) for k, v in row['Parameters'].items()}

    factor = res[res['Parameters'] == params]['Factor'].values[0]

    return (snp / factor) / (4 * 1 * mu)


def generate_data(params, model, nb_simu, path_data, path_length, typ):
    """
    Generate a set of data with msprime.
    """
    # Define length
    length = length_from_file(path_length, params, mu=8e-2, snp=10000) if typ == 'sfs' else 1e3

    # Convert params from log scale
    params.update({k: (np.power(10, v) if k != 'm21' else v) for k, v in params.items()})

    # Parameters for the simulation
    params.update(
        simulation_parameters(sample=20, ne=1, rcb_rate=8e-2, mu=8e-2, length=length))

    if typ == 'sfs':
        data = generate_sfs(params, model, nb_simu)

    else:
        data = generate_vcf(params)

    print("SNPs: {}".format(round(np.mean(data['SNPs'][0]))))
    
    # Export DataFrame to json file
    data.to_json(path_data)

    # Zip file
    f.zip_file(data=path_data)


# Generate SFS

def generate_sfs(params, model, nb_simu):
    """
    Generate a set of unfolded sfs of fixed SNPs size with msprime.
    """
    sfs, snp, execution = [], [], []
    for i in range(nb_simu):
        start_time = time.time()

        sfs_observed = ms.msprime_simulation(model=model, params=params)

        sfs.append(sfs_observed)
        snp.append(sum(sfs_observed))
        execution.append(time.time() - start_time)

    # Create DataFrame from dictionary
    dico = {
        'Parameters': [params], 'SNPs': [snp], 'SFS observed': [sfs],
        'Time': [round(np.mean(execution), 4)]
    }
    return pd.DataFrame(dico)


# Generate VCF

def generate_vcf(params):
    """
    Generate a set of unfolded sfs of fixed SNPs size with msprime.
    """
    start_time = time.time()

    sfs, variants = ms.msprime_simulate_variants(params, debug=True)
            
    # Create DataFrame from dictionary
    dico = {
        'Parameters': [params], 'SNPs': [sum(sfs)], 'SFS observed': [sfs],
        'Variants': [variants], 'Time': [time.time() - start_time]
    }
    return pd.DataFrame(dico)


######################################################################
# Grip point optimization                                            #
######################################################################

def grid_optimisation():
    """
    Define the optimal value of the smallest size of the grid point.
    """
    mu_list, log_scale = [2e-3, 4e-3, 8e-3, 12e-3], [1, 2, 4, 6, 10, 20, 40, 60]

    dico = {}

    for mu in mu_list:
        # Parameters for the simulation
        fixed_params = simulation_parameters(sample=20, ne=1, rcb_rate=mu, mu=mu, length=1e5)
        print("Msprime simulation - sample size {} & mutation rate {}".format(20, mu))

        # Msprime simulation
        sfs = ms.msprime_simulation(model=ms.constant_model, fixed_params=fixed_params)

        # Generate the SFS file compatible with dadi
        f.dadi_data(sfs, dadi.constant_model.__name__)

        # Data
        theoretical_theta = computation_theoretical_theta(ne=1, mu=mu, length=1e5)
        dico[mu] = {
            "Likelihood": [], "Estimated theta": [], "Theoretical theta": [theoretical_theta]*9
        }

        # Optimisation grid point for the extrapolation
        for _, scale in enumerate(log_scale):
            point = (len(sfs) - 1) * scale
            pts_list = [point, point+10, point+20]

            likelihood, estimated_theta, _ = dadi.dadi_inference(pts_list, dadi.constant_model)

            dico[mu]["Likelihood"].append(likelihood)
            dico[mu]["Estimated theta"].append(estimated_theta)

    plot.plot_optimisation_grid(dico, log_scale)


######################################################################
# Optimization of dadi parameters                                    #
######################################################################

def dadi_params_optimisation(sample):
    """
    Determine the error rate of the inference of 100 observed - simulated with msprime.

    Each observed is a constant population model. The goal is to determine the best mutation
    rate mu and the best number of sampled genomes n.

      - mu: the mutation rate
      - n: the number of sampled monoploid genomes

    Parameter
    ---------
    sample: int
        it's n
    """
    mu_list = [2e-3, 4e-3, 8e-3, 12e-3, 2e-2]  # ,  8e-2,  2e-1]
    nb_simu = 3

    # Grid point for the extrapolation
    pts_list = [sample*10, sample*10 + 10, sample*10 + 20]

    # Set up the Pandas DataFrame
    col = ["Theoretical theta", "Error rate", "mu"]
    data = pd.DataFrame(columns=col)

    # List of execution time of each simulation
    execution_time = []

    for mu in mu_list:
        tmp = []

        # Parameters for the simulation
        params = simulation_parameters(sample=sample, ne=1, rcb_rate=mu, mu=mu, length=1e5)
        print("Msprime simulation - sample size {} & mutation rate {}".format(sample, mu))

        for i in range(nb_simu):
            start_time = time.time()
            print("Simulation: {}/{}".format(i+1, nb_simu), end="\r")

            # Simulation for a constant population with msprime
            sfs = ms.msprime_simulation(model=ms.constant_model, param=params)

            # Generate the SFS file compatible with dadi
            f.dadi_data(sfs, dadi.constant_model.__name__, path="./Data/Error_rate/",
                        name="SFS-{}".format(sample))

            # Dadi inference
            _, estimated_theta, _ = dadi.dadi_inference(
                pts_list, dadi.constant_model, path="./Data/Error_rate/",
                name="SFS-{}".format(sample)
            )

            theoretical_theta = computation_theoretical_theta(ne=1, mu=mu, length=1e5)
            error_rate = estimated_theta / theoretical_theta

            row = {
                "Theoretical theta": theoretical_theta, "Error rate": error_rate, "mu": mu
            }
            data = data.append(row, ignore_index=True)

            tmp.append(time.time() - start_time)

        # Mean execution time for the 100 simulation with the same genome length and mu
        mean_time = round(sum(tmp) / nb_simu, 3)
        execution_time.extend([mean_time for _ in range(nb_simu)])

        data["Execution time"] = execution_time

    # Delete sfs file
    os.remove("./Data/Error_rate/SFS-{}.fs".format(sample))

    # Export data to csv file
    data.to_csv("./Data/Error_rate/error-rate-{}.csv".format(sample), sep='\t', index=False)


######################################################################
# SFS evaluation                                                     #
######################################################################

def compute_weighted_square_distance(data):
    """
    Compute the weighted square distance between the theoretical SFS and the observed ones.
    """
    theoritical_sfs = plot.normalization(plot.compute_theoretical_sfs(19))
    
    d2 = {}
    for i, row in data.iterrows():
        d2[i] = []
        for sfs in row['SFS observed']:
            d2[i].append(sum(
                [
                    np.power(eta_obs - eta_theo, 2) / eta_theo for eta_obs, eta_theo in
                    zip(plot.normalization(sfs), theoritical_sfs)
                ]
            ))

        d2[i] = np.mean(d2[i])

    return d2


######################################################################
# Inference with Dadi                                                #
######################################################################

def likelihood_ratio_test(ll_m0, ll_m1, dof):
    """
    Likelihood-ratio test to assesses the godness fit of two model.

    c.f. jupyter notebook "analyse.ipynb" for more information

    Parameters
    ----------
    ll_m0: float
        log-likelihood of model m0
    ll_m1: float
        log-likelihood of model m1
    dof: int
        degree of freedom

    Return
    ------
    Either 1 - test significant and reject of H0
        Or 0 - test insignificant and no reject of H0
    """
    lrt = 2 * (ll_m1 - ll_m0)  # LL ratio test
    p_value = chi2.sf(lrt, dof)  # Chi2 test

    if p_value > 0.05:
        return 0  # test insignificant and no reject of H0
    return 1  # test significant and reject of H0


def weighted_square_distance(sfs):
    """
    Compute the weighted square distance d2.

    c.f. jupyter notebook "analyse.ipynb" for more information

    Parameter
    ---------
    sfs: dictionary
        Either observed SFS and inferred SFS with M1
            Or inferred SFS of two models - M0 & M1

    Return
    ------
    d2: float
        the weighted square distance
    """
    # Normalization of the SFS
    normalized_sfs = {}
    for key, spectrum in sfs.items():
        normalized_sfs[key] = [ele / sum(spectrum) for ele in spectrum]

    # Weighted square distance
    if "Observed" in sfs.keys():
        d2 = [
            np.power(eta_model - eta_obs, 2) / eta_model for eta_obs, eta_model in
            zip(normalized_sfs['Observed'], normalized_sfs['Model'])
        ]
    else:
        d2 = [
            np.power(eta_m0 - eta_m1, 2) / (np.mean([eta_m0, eta_m1])) for eta_m0, eta_m1 in
            zip(normalized_sfs['M0'], normalized_sfs['M1'])
        ]

    del normalized_sfs

    return sum(d2)


def compute_dadi_inference(sfs_observed, models, sample, fold, path_data, job, dof, fixed,
                           value):
    """
    Parameter
    ---------
    sfs_observed: list
        the observed SFS generated with msprime
    models: dictionary
      - Inference
        The model with more parameters, i.e. M1
      - Control
        The model with less parameters, i.e. M0
    sample: int
        The number of sampled monoploid genomes
    fixed: str
        fixed parameter for the inference, either (tau), (kappa) or (migr)
    dof: int
        degrees of freedom

    Return
    ------
    data: dictionary
      - LRT
        Likelihood ratio test
      - M0
        List of log likelihood and sfs for the inference with M0
      - M1
        List of log likelihood, sfs and estimated parameters for the inference with M1.
        In this case from the same observed SFS, 1000 inferences with M1 are made and only the
        best one is kept. I.E. the one with the highest log-likelihood.
      - Time
        Mean execution time for the inference
      - d2 observed inferred
        Weighted square distance between observed SFS & inferred SFS with M1
      - d2 models
        Weighted square distance between inferred SFS with M0 & M1
    """
    data = {
        'LRT': [], 'M0': {'LL': [], 'SFS': []}, 'M1': {'LL': [], 'SFS': [], 'Estimated': []},
        'Time': 0, 'd2 observed inferred': [], 'd2 models': []
    }
    execution = []

    # Grid point for the extrapolation
    if models['Inference'].__name__.split('_', 1)[0] == 'twopops':  # migration
        pts_list = [round(sample/2), round(sample/2) + 10, round(sample/2) + 20]
    else:  # suddden decline or growth
        pts_list = [sample*10, sample*10 + 10, sample*10 + 20]

    for i, sfs in enumerate(sfs_observed):
        print("SFS observed {}".format(i))
        # Generate the SFS file compatible with dadi
        dadi_file = "SFS-{}".format(job) if value is None else "SFS_{}-{}".format(value, job)
        f.dadi_data(sfs, models['Inference'].__name__, fold, path=path_data, name=dadi_file)

        # Dadi inference for M0
        # Pairs (Log-likelihood, Inferred SFS)
        m0_inference = dadi.inference(pts_list, models['Control'], path=path_data,
                                      name=dadi_file)
        data['M0']['LL'].append(m0_inference[0])
        data['M0']['SFS'].append(m0_inference[1])

        # Dadi inference for M1
        m1_inferences, m1_execution = [], []

        for _ in range(2):  # Run 100 inferences with dadi from the observed sfs
            start_inference = time.time()

            # Pairs (Log-likelihood, Inferred SFS, Params)
            tmp = dadi.inference(pts_list, models['Inference'], fixed=fixed, value=value,
                                 path=path_data, name=dadi_file, verbose=True)

            m1_inferences.append(tmp)
            m1_execution.append(time.time() - start_inference)

        execution.append(np.mean(m1_execution))

        # m1_inferences is a list of pairs (Log-likelihood, Inferred SFS, Params)
        # Compare each item of this list by the value at index 0, i.e. the log-likelihood and
        # select the one with this highest value.
        index_best_ll = m1_inferences.index((max(m1_inferences, key=lambda ele: ele[0])))

        data['M1']['LL'].append(m1_inferences[index_best_ll][0])
        data['M1']['SFS'].append(m1_inferences[index_best_ll][1])
        data['M1']['Estimated'].append(m1_inferences[index_best_ll][2])

        # Compute the log-likelihood ratio test between M0 and M1
        data['LRT'].append(
            likelihood_ratio_test(data['M0']['LL'][i], data['M1']['LL'][i], dof)
        )

        # Compute weighted square distance
        data['d2 observed inferred'].append(
            weighted_square_distance({'Observed': sfs, 'Model': data['M1']['SFS'][i]})
        )  # d2 between the observed SFS & inferred SFS with M1

        data['d2 models'].append(
            weighted_square_distance({'M0': data['M0']['SFS'][i], 'M1': data['M1']['SFS'][i]})
        )  # d2 between the inferred SFS of two models - M0 & M1

        if i == 1:
            break

    # Mean execution time for the inference
    data['Time'] = round(sum(execution) / len(sfs_observed), 4)

    return data


def save_dadi_inference(simulation, models, fold, path_data, job, fixed, value):
    """
    Inference with dadi.

    Parameter
    ---------
    simulation: dictionary
      - Parameters
        Parameters for the simulation with msprime - mutation rate mu, recombination rate, Ne,
        length L of the sequence, sample size.
      - SNPs
        List of SNPs for each observed SFS
      - SFS observed
        List of the observed SFS generated with msprime for the same set of parameters

    models: dictionary
      - Inference: the custom model to infer demography history, i.e. the model m1 with more
        parameters
      - Control: the control model, i.e. the model m0 with less parameters

    fixed
        fixed parameter for the inference, either (tau), (kappa) or (migration)
    value
        Value of the fixed parameters for the inference - log scale
    """
    # Inference
    sfs_observed, sample = simulation['SFS observed'], simulation['Parameters']['sample_size']

    if value is None:
        inf = compute_dadi_inference(sfs_observed, models, sample, fold, path_data, job, dof=2,
                                     fixed=fixed, value=value)
    else:
        inf = compute_dadi_inference(sfs_observed, models, sample, fold, path_data, job, dof=2,
                                     fixed=fixed, value=np.power(10, value))

    # Save data
    params = {
        k: v for k, v in simulation['Parameters'].items() if k in ['Tau', 'Kappa', 'm12',
                                                                   'm21']
    }
    params['Theta'] = 4 * 1 * 8e-2 * simulation['Parameters']['length']  # 4 * Ne * mu * L

    # Create DataFrame from dictionary
    dico = {
        'Parameters': [params], 'Positive hit': [sum(inf['LRT'])],
        'SNPs': [simulation['SNPs']], 'SFS observed': [sfs_observed], 'M0': [inf['M0']],
        'M1': [inf['M1']], 'Time': [inf['Time']],
        'd2 observed inferred': [np.mean(inf['d2 observed inferred'])],
        'd2 models': [np.mean(inf['d2 models'])]
    }
    data = pd.DataFrame(dico)

    # Export dataframe to json files
    if fixed is None:
        name = "dadi_{}-{}".format(models['Inference'].__name__.split('_')[1], job)
    else:
        name = "dadi_{}_{}={}-{}" \
            .format(models['Inference'].__name__.split('_')[1], fixed, value, job)
    data.to_json("{}{}".format(path_data, name))

    # Zip file
    f.zip_file(data="{}{}".format(path_data, name))

    # Remove SFS file
    if value is None:
        os.remove("{}SFS-{}.fs".format(path_data, job))
    else:
        os.remove("{}SFS_{}-{}.fs".format(path_data, np.power(10, value), job))


######################################################################
# Optimization of inference with Dadi                                #
######################################################################

def data_optimization_dadi(model, filout, path_length):
    """
    Generate the data for the optimization of dadi for various SNPs' values - from 1e4 to 5e5 
    """
    # Set up (Tau, Kappa) & length
    if model == 'decline':  # sudden decline
        params = {'Tau': 0., 'Kappa': 1.}
    elif model == 'growth':  # sudden growth
        params = {'Tau': 0., 'Kappa': -1.}
    else:  # constant
        params = {'Tau': 0., 'Kappa': 0.}  # Constant

    # Define length
    snp = float(filout.split('=')[1])
    length = length_from_file(path_length, params, mu=8e-2, snp=snp)

    # Convert params from log scale
    params.update({k: np.power(10, v) for k, v in params.items()})

    # Parameters for the simulation
    params.update(
        simulation_parameters(sample=20, ne=1, rcb_rate=8e-2, mu=8e-2, length=length)
    )

    # Generation of data
    data = generate_sfs(params, model=ms.sudden_decline_model, nb_simu=1)

    # DataFrame to json
    data.to_json(filout)

    # Zip file
    f.zip_file(filout)


def compute_optimization_dadi(filin, path_data, models, job):
    """
    Optimization of inference with SMC++ with various SNPs for simple scenario:

      - Sudden decline with tau = 0 & kappa = 1, decline of force 10 at a time 1 in the past
      - Sudden growth with tau = 0 & kappa = -1, growth of force 10 at a time 1 in the past
      - Constant model with kappa = 0, so there are no change in the population size in the past

    Important
    Each value of tau & kappa are given in log scale.
    """
    # Load data
    simulation = pd.read_json(filin).iloc[0]

    # Set up file
    filout = "{}{}".format(path_data, filin.rsplit('/', 1)[1].rsplit('.', 1)[0])

    # Inference with dadi
    sfs_observed, sample = simulation['SFS observed'], simulation['Parameters']['sample_size']

    inf = compute_dadi_inference(
        sfs_observed, models, sample, fold=False, path_data=path_data, job=job, dof=2,
        fixed=None, value=None
    )

    # Save data
    params = {
        k: v for k, v in simulation['Parameters'].items() if k in ['Tau', 'Kappa', 'm12',
                                                                   'm21']
    }
    params['Theta'] = 4 * 1 * 8e-2 * simulation['Parameters']['length']  # 4 * Ne * mu * L

    # Create DataFrame from dictionary
    dico = {
        'Parameters': [params], 'Positive hit': [sum(inf['LRT'])],
        'SNPs': [simulation['SNPs']], 'SFS observed': [sfs_observed], 'M0': [inf['M0']],
        'M1': [inf['M1']], 'Time': [inf['Time']],
        'd2 observed inferred': [np.mean(inf['d2 observed inferred'])],
        'd2 models': [np.mean(inf['d2 models'])]
    }
    data = pd.DataFrame(dico)

    # Export DataFrame to json
    data.to_json("{}".format(filout))

    # Zip file
    f.zip_file(data="{}".format(filout))

    # Remove SFS file
    os.remove("{}SFS-{}.fs".format(path_data, job))


######################################################################
# Inference with stairway plot 2                                     #
######################################################################

def compute_stairway_inference(simulation, path_stairway, path_data, fold):
    """
    Parameter
    ---------
    path_stairway: path to the folder stairway_plot__es
    path_data: path to the blueprint file
    """
    # Data
    stairway = pd.DataFrame()

    tau_list, kappa_list = [-4., 0., 2.4], [-3.5, 0., 2.9]
    if 'Tau' in simulation['Parameters']:
        tau = round(np.log10(simulation['Parameters']['Tau']), 2)
        kappa = round(np.log10(simulation['Parameters']['Kappa']), 2)
    elif 'm12' in simulation['Parameters']:
        tau = round(np.log10(simulation['Parameters']['m12']), 2)
        kappa = round(np.log10(simulation['Parameters']['Kappa']), 2)
    else:
        tau = None

    # Inference - the inference is only done with one of the observed SFS among the 100 for a
    # matter of efficiency
    all_sfs = simulation['SFS observed']
    sfs = np.array([sum(spectrum) for spectrum in zip(*all_sfs)]) / len(all_sfs)

    blueprint = "stairway_inference"

    # Generate the SFS file compatible with stairway plot v2
    data = {
        k: v for k, v in simulation['Parameters'].items() if k in ['sample_size', 'mu']
    }
    data['sfs'], data['year'], data['ninput'] = sfs, 1, 200

    f.stairway_data(blueprint, data, path_data, fold)

    # Create the batch file
    os.system("java -cp {0}stairway_plot_es Stairbuilder {1}{2}.blueprint"
              .format(path_stairway, path_data, blueprint))

    # Run the batch file
    os.system("xvfb-run -a bash {}{}.blueprint.sh".format(path_data, blueprint))

    # Extract data from the inference with stairway

    # M0: default model | M1: the model with 2 dimension | Final: the final model
    dico = f.read_stairway_final("{}{}/final/".format(path_data, blueprint))

    # Ne: pair (Ne min, Ne max) | Year: pair (Year of Ne min, Year of Ne max)
    dico.update(f.read_stairway_summary("{0}{1}/{1}.final.summary".format(path_data,
                                                                          blueprint)))

    # keep track of parameters
    if 'Kappa' in simulation['Parameters'].keys():
        dico['Parameters'] = {
            k: v for k, v in simulation['Parameters'].items() if k in ['Tau', 'Kappa', 'm12']
        }
    else:
        dico['Parameters'] = simulation['Parameters']['Ne']

    # Summarize the data of the inference
    stairway = stairway.append(dico, ignore_index=True)

    # Keep track of some figure generated by stairway plot
    if tau in tau_list and kappa in kappa_list:
        figure = "{0}{1}/{1}.final.summary.png".format(path_data, blueprint)
        os.system("mv {} ./Figures/Stairway/{}".format(figure, path_data.rsplit('/', 2)[1]))

    # Remove all blueprint file & stairway file
    # os.system("rm -rf {}".format(path_data))

    return stairway


def save_stairway_inference(simulation, model, fold):
    """
    Inference with stairway plot 2.

    Parameter
    ---------
    simulation: dictionary
      - Parameters
        Parameters for the simulation with msprime - mutation rate mu, recombination rate, Ne,
        length L of the sequence, sample size.
      - SFS observed
        List of the observed SFS generated with msprime for the same set of parameters

    model: str
        either decline, migration or cst
    """
    # Set up path data
    path_stairway = "/home/damase/All/Cours/M2BI-Diderot/Species_evolution_inference/sei/" \
        "inference/stairway_plot_v2.1.1/"

    if model == 'decline':
        param = {k: round(np.log10(v), 2) for k, v in simulation['Parameters'].items()
                 if k in ['Tau', 'Kappa']}
        file_data = "stairway_{}_tau={}_kappa={}".format(model, param['Tau'], param['Kappa'])

    elif model == 'migration':
        param = {k: round(np.log10(v), 2) for k, v in simulation['Parameters'].items()
                 if k in ['m12', 'Kappa']}
        file_data = "stairway_{}_m12={}_kappa={}".format(model, param['m12'], param['Kappa'])

    else:
        file_data = "stairway_{}-ne={}/".format(model, simulation['Parameters']['Ne'])

    file_data += "_folded" if fold else "_unfolded"
    path_data = path_stairway + file_data + "/"
    
    if not os.path.isdir(path_data):
        os.mkdir(path_data)
        os.system("cp -r {} {}".format(path_stairway + "stairway_plot_es", path_data))

    # Compute the inference with stairway plot 2
    data = compute_stairway_inference(simulation, path_stairway, path_data, fold)

    # Convert pandas DataFrame data to json file
    path_data = "./Data/Stairway/{}/".format(model)
    path_data += "Folded/" if fold else "Unfolded/"
    data.to_json("{}{}".format(path_data, file_data))

    # Zip file
    f.zip_file(data="{}{}".format(path_data, file_data))


######################################################################
# Inference with SMC++                                               #
######################################################################

def compute_smc_inference(simulation, param, filout, path_data):
    """
    Inference with SMC++:
      - Generation of the data in the format compatible with the SMC++ software.
      - Perform the inference
    """
    # Generate the VCF file format
    f.variants_to_vcf(simulation['Variants'], param, filout, path_data, ploidy=2)

    # VCF file to SMC++ format
    f.vcf_to_smc(filout, path_data)

    # Estimation
    smc_estimate = (
        "smc++ estimate --em-iterations 1 -o {0}{1} --knots 8 {2} {0}smc_{1}.gz"
    ).format(path_data, filout.split('_', 1)[1], 8e-4)
    os.system(smc_estimate)

    # Plot
    smc_plot = (
        "smc++ plot -c {0}{1}/{1}.png {0}{1}/model.final.json"
    ).format(path_data, filout.split('_', 1)[1])
    os.system(smc_plot)

    # Remove vcf, smc and index file
    os.system("rm -rf {0}*{1}.gz*".format(path_data, filout.split('_', 1)[1]))


def save_smc_inference(simulation, model):
    """
    Inference with SMC++.

    Parameter
    ---------
    simulation: dictionary
      - Parameters
        Parameters for the simulation with msprime - mutation rate mu, recombination rate, Ne,
        length L of the sequence, sample size.
      - Variants
        List of position and genotype for each variant with 0 the ancestral state and 1 the
        alternative one.

    model: str
        either decline, migration or cst
    """
    # Set up path data
    path_data = "./Data/SMC/{}/".format(model)

    if model == 'decline':
        param = {k: round(np.log10(v), 2) for k, v in simulation['Parameters'].items()
                 if k in ['Tau', 'Kappa']}
        filout = "vcf_tau={}_kappa={}".format(param['Tau'], param['Kappa'])

    elif model == 'migration':
        param = {k: round(np.log10(v), 2) for k, v in simulation['Parameters'].items()
                 if k in ['m12', 'Kappa']}
        filout = "vcf_m12={}_kappa={}".format(param['m12'], param['Kappa'])

    else:
        filout = "vcf_ne={}".format(simulation['Parameters']['Ne'])

    # Select sample size and length
    param = {
        k: v for k, v in simulation['Parameters'].items() if k in ['sample_size', 'length']
    }

    # Inference
    for i in range(1):
        print("Simulation: {}/1".format(i+1))
        compute_smc_inference(simulation, param, filout, path_data)


######################################################################
# Optimization of inference with SMC++                               #
######################################################################

def data_optimization_smc(model, filout):
    """
    Generate the data for the optimization of smc for various sequence length - from 1e2 to 5e6.
    """
    # Set up (Tau, Kappa) & length
    if model == 'decline':  # sudden decline
        params = {'Tau': 0., 'Kappa': 1.}
    elif model == 'growth':  # sudden growth
        params = {'Tau': 0., 'Kappa': -1.}
    else:  # constant
        params = {'Tau': 0., 'Kappa': 0.}  # Constant

    # Convert params from log scale
    params.update({k: np.power(10, v) for k, v in params.items()})

    # Parameters for the simulation
    length = float(filout.split('=')[1])
    params.update(
        simulation_parameters(sample=20, ne=1, rcb_rate=8e-2, mu=8e-2, length=length)
    )

    # Generation of data
    sfs, variants = ms.msprime_simulate_variants(params, debug=True)

    # Save data
    dico = {
        'Parameters': [params], 'SNPs': [sum(sfs)], 'SFS observed': [sfs], 'Variants': [variants]
    }
    data = pd.DataFrame(dico)

    # DataFrame to json
    data.to_json(filout)

    # Zip file
    f.zip_file(filout)

    
def compute_optimization_smc(filin, path_data):
    """
    Optimization of inference with SMC++ with various sequence length and SNPs for simple
    scenario:
      - Sudden decline with tau = 0 & kappa = 1, decline of force 10 at a time 1 in the past
      - Sudden growth with tau = 0 & kappa = -1, growth of force 10 at a time 1 in the past
      - Constant model with kappa = 0, so there are no change in the population size in the past

    Important
    Each value of tau & kappa are given in log scale.

    For each data, various inference are done with knot value from 2 to 8.
    """
    # Load data
    data = pd.read_json(filin).iloc[0]

    # Set up file & folder
    filout = "vcf_length={:.1e}".format(data['Parameters']['length'])
    folder = "{}{}".format(path_data, filout.split('_')[1])

    if os.path.isdir(folder):
        os.system("rm -rf {}".format(folder))
    os.mkdir(folder)

    # Variants to VCF format file
    f.variants_to_vcf(
        variants=data['Variants'], param=data['Parameters'], fichier=filout, path_data=path_data
    )

    # VCF to SMC++ file
    f.vcf_to_smc(fichier=filout, path_data=path_data)

    # Inference for various knot value
    for knot in [2, 3, 4, 5, 6, 7, 8]:
        print("\n\n\n###\nKnot {}\n###\n\n".format(knot))

        # Estimation
        smc_estimate = (
            "smc++ estimate --em-iterations 1 -o {0}.{1}-KNOTS={2}/ --knots {2} {3} "
            "{0}smc_{1}.gz"
        ).format(path_data, filout.split('_')[1], knot, 8e-4)
        os.system(smc_estimate)

        # Plot
        smc_plot = (
            "smc++ plot -c {0}{1}/plot_knot={2}.png {0}.{1}-KNOTS={2}/model.final.json"
        ).format(path_data, filout.split('_')[1], knot)
        os.system(smc_plot)

        break

    # Remove vcf, smc and index file
    os.system("rm -rf {}*{}.gz*".format(path_data, filout.split('_')[1]))


######################################################################
# Main                                                               #
######################################################################

def main():
    """
    Le main du programme.
    """
    # grid_optimisation()

    args = arg.arguments()

    if args.analyse == 'data':

        # Simulation of constant population - inference of decline population
        if args.model == 'cst':
            # Parameters for the simulation
            params = simulation_parameters(sample=20, ne=1, rcb_rate=8e-2, mu=8e-2, length=1e5)

            dico = {
                'Parameters': params, 'SFS observed': [], 'SNPs': [], 'Time': [], 'Variants': []
            }
            for i in range(1):
                print("Simulation {}/100".format(i+1), end="\r")
                start_time = time.time()

                if i == 0:
                    sfs_cst, variants = \
                        ms.msprime_simulation(model=ms.constant_model, params=params)
                    dico['Variants'].append(variants)  # = variants
                else:
                    sfs_cst, _ = \
                        ms.msprime_simulation(model=ms.constant_model, params=params)

                dico['SFS observed'].append(sfs_cst)
                dico['SNPs'].append(sum(sfs_cst))

                dico['Time'].append(time.time() - start_time)

            # Compute mean execution time
            dico['Time'] = round(np.mean(dico['Time']), 4)

            # Store data in pandas DataFrame
            simulation = pd.DataFrame(columns=['Parameters', 'SFS observed', 'SNPs', 'Time'])
            simulation = simulation.append(dico, ignore_index=True)

            # Export to json
            data = "./Data/Msprime/cst/SFS_{}_Ne={}".format(args.model, params['Ne'])
            simulation.to_json(data)
            f.zip_file(data)
            print("Over")
            sys.exit()

        # Generate length factor file
        elif args.file:
            path_data = "./Data/Msprime/{}/".format(args.model)
            filin = "SFS_{}-all".format(args.model)

            # Export the observed SFS to DataFrame
            simulation = f.export_simulation_files(filin, path_data)

            factor, theta = pd.DataFrame(columns=['Parameters', 'Factor']), 32000
            if args.model == 'decline':
                # Convert Tau & kappa to log10
                factor['Parameters'] = simulation['Parameters'].apply(lambda ele: {
                    'Tau': round(np.log10(ele['Tau']), 2),
                    'Kappa': round(np.log10(ele['Kappa']), 2)
                })

            else:
                # Convert m12 & kappa to log10
                factor['Parameters'] = simulation['Parameters'].apply(lambda ele: {
                    'm12': round(np.log10(ele['m12']), 2), 'm21': ele['m21'],
                    'Kappa': round(np.log10(ele['Kappa']), 2)
                })

            # Compute mean SNPs
            factor['Factor'] = simulation['SNPs'].apply(lambda snp: np.mean(snp) / theta)

            # Export pandas DataFrame factor to json file
            factor.to_json('./Data/Msprime/length_factor-{}'.format(args.model))

            sys.exit()

        # Simulation of sudden decline model with msprime for various tau & kappa
        if args.model == 'decline':
            params = define_parameters(args.model, args.typ)
            params, model = params[args.job-1], ms.sudden_decline_model
            path_data = (
                "./Data/Msprime/{0}/{3}_{0}_tau={1}_kappa={2}"
            ).format(args.model, params['Tau'], params['Kappa'], args.typ.upper())

        # Simulation of two populations migration models for various migration into 1 from
        # 2 (with m12 the migration rate) and no migration into 2 from 1
        # Population 1 size is pop1 and population 2 size is pop2 = kappa*pop1
        elif args.model == 'migration':
            params = define_parameters(args.model, args.typ)
            params, model = params[args.job-1], ms.twopops_migration_model

            path_data = (
                "./Data/Msprime/{0}/{3}_{0}_m12={1}_kappa={2}"
            ).format(args.model, params['m12'], params['Kappa'], args.typ.upper())

        path_length = "./Data/Msprime/length_factor-{}".format(args.model)

        generate_data(params, model, nb_simu=2, path_data=path_data, path_length=path_length,
                      typ=args.typ)

    elif args.analyse == 'opt':
        dadi_params_optimisation(args.number)

    elif args.analyse == 'inf':
        # Export the observed data to DataFrame
        typ = 'VCF' if args.smc else 'SFS'
        path_sim = "./Data/Msprime/{}/".format(args.model)
        simulation = f.export_simulation_files(typ=typ, path_data=path_sim, job=args.job - 1,
                                               param=args.param, value=args.value)

        # Inference with dadi
        if args.dadi:

            # Set up M0 & M1 model for the inference with dadi
            if args.model == "decline":
                models = \
                    {'Inference': dadi.sudden_decline_model, 'Control': dadi.constant_model}
            elif args.model == "migration":
                models = \
                    {'Inference': dadi.twopops_migration_model, 'Control': dadi.constant_model}

            if args.param is None:
                path_inf = "./Data/Dadi/{}/all/".format(args.model)
            else:
                path_inf = "./Data/Dadi/{}/{}/".format(args.model, args.param)

            path_inf += "Folded/" if args.fold else "Unfolded/"

            save_dadi_inference(simulation, models, args.fold, path_inf, args.job,
                                fixed=args.param, value=args.value)

        # Inference with stairway plot 2
        elif args.stairway:
            save_stairway_inference(simulation, model=args.model, fold=args.fold)

        # Inference with SMC++
        elif args.smc:
            save_smc_inference(simulation, model=args.model)

    elif args.analyse == 'optsmc':
        length = [1e2, 2.5e4, 5e4, 7.5e4, 1e5, 2.5e5, 5e5, 7.5e5, 1e6, 2.5e6, 5e6][args.job-1]

        # Set up path and file
        path_data = "./Data/SMC/optimization_smc/{}/".format(args.model)
        filout = (
            "./Data/SMC/optimization_smc/data/vcf_{}_length={:.1e}"
        ).format(args.model, length)

        if "{}.zip".format(filout.rsplit('/', 1)[1]) \
           not in os.listdir("./Data/SMC/optimization_smc/data/"):
            # Generate data
            data_optimization_smc(args.model, filout)

        compute_optimization_smc(filin="{}.zip".format(filout), path_data=path_data)

    elif args.analyse == 'optdadi':
        snps = [1e4, 2.5e4, 5e4, 7.5e4, 1e5, 2e5, 3e5, 4e5, 5e5][args.job-1]

        # Set up path and file
        path_data = "./Data/Dadi/optimization_dadi/{}/".format(args.model)
        filout = (
            "./Data/Dadi/optimization_dadi/data/sfs_{}_snps={:.1e}"
        ).format(args.model, snps)

        path_length = "./Data/Msprime/length_factor-decline"

        if "{}.zip".format(filout.rsplit('/', 1)[1]) \
           not in os.listdir("./Data/Dadi/optimization_dadi/data/"):
            # Generate data
            data_optimization_dadi(args.model, filout, path_length)

        models = \
            {'Inference': dadi.sudden_decline_model, 'Control': dadi.constant_model}

        compute_optimization_dadi(
            filin="{}.zip".format(filout), path_data=path_data, models=models, job=args.job
        )

    elif args.analyse == 'optsnp':
        snps = [1e4, 2e4, 5e4, 7.5e4, 1e5, 2e5, 3e5, 4e5, 5e5]
        tmp = define_parameters(model='decline', typ='sfs')[args.job-1]

        # Set up path and file
        filout = (
            "./Data/optimization_snps/sfs_tau={}_kappa={}"
        ).format(tmp['Tau'], tmp['Kappa'])
        path_length = "./Data/Msprime/length_factor-decline"

        data = pd.DataFrame()
        for snp in snps:
            # Define length
            length = length_from_file(path_length, tmp, mu=8e-2, snp=snp)

            # Convert params from log scale
            params = ({k: np.power(10, v) for k, v in tmp.items()})

            # Parameters for the simulation
            params.update(
                simulation_parameters(sample=20, ne=1, rcb_rate=8e-2, mu=8e-2, length=length)
            )

            # Generation of data
            simulation = generate_sfs(params, model=ms.sudden_decline_model, nb_simu=1)
            data = data.append(simulation, ignore_index=True)

        # DataFrame to json
        data.to_json(filout)

        # Zip file
        f.zip_file(filout)


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()
