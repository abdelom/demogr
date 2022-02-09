"""
Analysis of inference with stairway plot 2.
"""

import pandas as pd
import numpy as np
import sys

from ..sei import likelihood_ratio_test


def stairway_ll_test(data, model):
    """
    Compute the log likelihood ratio between two models.

      - Either M0 & M1
      -     Or M1 & the final model

    Parameter
    ---------
    data: pandas DataFrame
    model: either m1 or final

    Return
    ------
    df: pandas DataFrame
      - Tau
      - Kappa
      - Positive hit
    """
    key = [param for param in data.iloc[0]['Parameters'].keys()]
    df = pd.DataFrame(columns=[key[0], key[1], 'Positive hit'])

    # Pre-processing data
    for _, row in data.iterrows():
        # Extract parameters use to generate the observed SFS
        # Then compute log10 of these parameters
        dico = {}
        for param in row['Parameters'].keys():
            dico[param] = round(np.log10(row['Parameters'][param]), 2)

        # Compute log-likelihood ratio test
        # For some inference there are only 1 dimension, in this case the LL of M1 is None
        if model == 'm1':
            lrt = [
                likelihood_ratio_test(ll_m0, ll_m1, dof=2) if ll_m1 is not None else 0
                for ll_m0, ll_m1 in zip(row['M0']['LL'], row['M1']['LL'])
            ]
        else:
            dimensions = [len(ele) for ele in row['Final']['Theta']]
            lrt = [
                likelihood_ratio_test(ll_m1, ll_final, dof=dof) if ll_m1 is not None else 0
                for ll_m1, ll_final, dof in zip(row['M1']['LL'], row['Final']['LL'], dimensions)
            ]

        dico['Positive hit'] = (sum(lrt) / 200) * 100 # pourcentage

        # Add to pandas DataFrame df
        df = df.append(dico, ignore_index=True)

    return df


def stairway_distance_ne(data):
    """
    Compute the distance between the minimum and maximum Ne.
    """
    key = [param for param in data.iloc[0]['Parameters'].keys()]
    df = pd.DataFrame(columns=[key[0], key[1], 'Ne'])

    # Pre-processing data
    for _, row in data.iterrows():
        # Extract parameters use to generate the observed SFS
        # Then compute log10 of these parameters
        dico = {}
        for param in row['Parameters'].keys():
            dico[param] = round(np.log10(row['Parameters'][param]), 2)
        
        # Compute distance - (max - min)**2 / max
        dico['Ne'] = np.log10(np.power(row['Ne'][1] - row['Ne'][0], 2) / row['Ne'][1])
        
        # Add to pandas DataFrame df
        df = df.append(dico, ignore_index=True)

    return df


def stairway_dimension_comparaison(data):
    """
    Compute the difference between m1's dimension (2) and final model's dimension (compute by
    stiarway plot).
    """
    key = [param for param in data.iloc[0]['Parameters'].keys()]
    df = pd.DataFrame(columns=[key[0], key[1], 'Dimensions'])

    # Pre-processing data
    for _, row in data.iterrows():
        # Extract parameters use to generate the observed SFS
        # Then compute log10 of these parameters
        dico = {}
        for param in row['Parameters'].keys():
            dico[param] = round(np.log10(row['Parameters'][param]), 2)
        
        # Compute log-likelihood ratio test
        # For some inference there are only 1 dimension, in this case the LL of M1 is None
        dim_final = [len(ele) for ele in row['Final']['Theta']]
        dim_m1 = 2
        
        dimensions = [
            -1 if ll_m1 is None else dim - dim_m1 for ll_m1, dim in 
            zip(row['M1']['LL'], dim_final)
        ]
        
        dico['Dimensions'] = np.mean(dimensions)
        
        # Add to pandas DataFrame df
        df = df.append(dico, ignore_index=True)
        
    return df


def stairway_distance_param(data, parameter):
    key = [param for param in data.iloc[0]['Parameters'].keys()]
    df = pd.DataFrame(columns=[key[0], key[1], 'Distance'])

    # Pre-processing data
    for _, row in data.iterrows():
        # Extract parameters use to generate the observed SFS
        # Then compute log10 of these parameters
        dico = {}
        for param in row['Parameters'].keys():
            dico[param] = round(np.log10(row['Parameters'][param]), 2)
        
        # Compute estimated
        if parameter == 'kappa':
            # Ne ancestral: size of the population before the sudden change
            #   Ne initial: size of the population at time 0
            if dico["Kappa"] > 0:  # Decline
                estimated = row['Ne ancestral'] / row['Ne initial']
            else:  # Growth
                estimated = row['Ne initial'] / row['Ne ancestral']
        
        elif parameter == 'tau':
            # Year: pair of (Year of Ne min, Year of ne max)
            if dico["Kappa"] > 0:  # Décroissance
                # Max(Year of Ne min): time juste after the sudden growth
                # Min(Year of Ne max): time juste before the sudden growth
                estimated = max(row['Year'][0]) - min(row['Year'][1])
            else: # Croissance
                # Min(Year of Ne min): time juste before the sudden decline
                # Max(Year of Ne max): time juste after the sudden decline
                estimated = min(row['Year'][0]) - max(row['Year'][1])
                 
        # Compute distance - (estimated - observed)**2 / observed
        dico['Distance'] = np.log10(np.power(estimated - row['Parameters'][param], 2) \
                                    / row['Parameters'][param])

        # Add to pandas DataFrame df
        df = df.append(dico, ignore_index=True)
        
    return df


if __name__ == "__main__":
    sys.exit()
