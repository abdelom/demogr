"""
Analysis of inference with dadi.
"""

import pandas as pd
import numpy as np
import sys


def extract_parameters(data, key):
    """
    Extract from the pandas DataFrame data, the observed parameters key for each simulation and
    the estimated one of each inferrence (the mean of inferred key for the 100 inferrence).

    Return the parameters estimated and observed in log10 scale.

    Return
    ------
    parameters: dict
      - Observed: the observed parameters key of each simulation
      - Estimated: the estimated parameters key of each inferrence (the mean)
    """
    parameters = {'Observed': [], 'Estimated': []}

    for _, row in data.iterrows():
        parameters['Observed'].append(row['Parameters'][key])
        parameters['Estimated'].append([estimated[key] for estimated in row['M1']['Estimated']])

    return parameters


def compute_distance(data, key):
    """
    Compute the distance d between the observed and the estimated parameter

    With d = (false - true)^2 / true

      - true: the observed parameter (used to generate the data)
      - false: the estimated parameter (estimated by dadi)

    Parameter
    ---------
    data: pandas DataFrame of inference with Dadi
    key: the parameters to check - either Tau, Kappa, m12 or Theta
    """
    parameters = extract_parameters(data, key)

    distance = []
    for observed, estimated in zip(parameters['Observed'], parameters['Estimated']):
        distance.append(
            np.mean([np.log10(np.power(ele - observed, 2) / observed) for ele in estimated])
        )
    return distance


def data_for_heatmap(data, value):
    """
    Set up dadi DataFrame to plot the heatmap with:
      - Column: either Tau or m12
      - Row: Kappa
      - Value: value

    Parameter
    ---------
    data: pandas DataFrame of inference with Dadi
    value: either
      - d2 observed inferred
        if plotting the weighted square distance between observed and inferred model
      - d2 models
        if plotting the weighted square distance between m0 and m1
      - d2 observed theoretical
        if plotting the weighted square distance between observed SFS and the theoretical one
      - Positive hit
        if plotting the significant log-likelihood ratio-test
      - Distance
        if plotting the distance between the observed and the estimated parameters
    """
    df = pd.DataFrame()

    # Compute log10 of parameters
    for param in [key for key in data.iloc[0]['Parameters'] if key != 'Theta']:
        df[param] = data['Parameters'].apply(lambda ele: round(np.log10(ele[param]), 2))

    if value in ['Positive hit', 'Distance']:
        df[value] = data[value]
    else:
        df[value] = data[value].apply(np.log10)

    return df


if __name__ == "__main__":
    sys.exit()
