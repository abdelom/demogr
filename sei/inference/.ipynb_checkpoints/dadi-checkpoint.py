"""
This module allows the inference of demographic history of population with dadi.
"""

import sys

import numpy as np
import dadi

FIXED = None
VALUE = None


def constant_model(ns, pts):
    """
    Constant model, i.e. population size is constant - control scenario.

    It's the standard neutral model.

    Parameter
    ---------
    ns: int
        the number of sampled genomes in resulting spectrum
    pts: list
        the number of grid points to use in integration
    """
    # Define the grid we'll use
    grid = dadi.Numerics.default_grid(pts)

    # Define the phi_ancestral, i.e. phi for the equilibrium ancestral population
    phi_ancestral = dadi.PhiManip.phi_1D(grid)

    # Calculate the spectrum from phi
    sfs = dadi.Spectrum.from_phi(phi_ancestral, ns, (grid,))

    return sfs


def params_model(params):
    """
    Define parameters for the inference.

    There are 4 cases:
      - Fixed param: (Tau)
        The length of time ago at which the event occured
      - Fixed param: (Kappa)
        The decline force (sudden decline model) or the difference in size between population 2
        and 1 with Pop2 = Kappa * Pop1
      - Fixed param: (Migration)
        The migration rate from population 2 to 1
      - Parameters: (Kappa, Tau) or (Kappa, Migration)
        No fixed parameters

    In each case, there are two parameters but we can fixed one of this to 'help' dadi fot the
    inference.
    """
    global FIXED, VALUE
    if FIXED == 'tau':  # Fixed param: (Tau), the length of time ago at which the event occured
        return params, VALUE
    elif FIXED == 'kappa':  # Fixed param: (Kappa), the decline force
        return VALUE, params
    elif FIXED == 'm12':  # Fixed param: (m12), the migration rate into 1 from 2
        return params[0], VALUE
    else:
        return params


def sudden_decline_model(params, ns, pts):
    """
    Sudden decline model of the population.

    At time tau in the past, an equilibrium population of size nu*kappa undergoing a sudden
    decline, reaching size nu with kappa the decline force.

    Parameter
    ---------
    params
      - kappa: ratio of contemporary population's size to ancient population's size
      - tau: the length of time ago at which size change happened
    ns: int
        the number of sampled genomes in resulting spectrum
    pts: list
        the number of grid points to use in integration
    """
    # Params (Kappa, Tau)
    kappa, tau = params_model(params)

    # Define the grid we'll use
    grid = dadi.Numerics.default_grid(pts)

    # Define the phi_ancestral, i.e. phi for the equilibrium ancestral population
    phi_ancestral = dadi.PhiManip.phi_1D(grid, nu=1.0*kappa)

    # Define the sudden decline event at a time tau in past
    phi = dadi.Integration.one_pop(phi_ancestral, grid, tau, nu=1.0)

    # Calculate the spectrum from phi
    sfs = dadi.Spectrum.from_phi(phi, ns, (grid,))

    return sfs


def twopops_migration_model(params, ns, pts):
    """
    Two populations migration model.

    In this model, there are:
      - Two populations with population 1 of size p1 and population 2 of size p2 = kappa*p1
      - Some migrations into population 1 from 2 with m12 the migration rate
      - No migrations into population 2 to 1, so m21 = 0.0

    Parameter
    ---------
    params
      - kappa: ratio of population's 1 size to population's 2 size
      - m12: migration rate into population 1 from 2
      - m21: migration rate into population 2 from 1
    ns: int
        the number of sampled genomes in resulting spectrum
    pts: list
        the number of grid points to use in integration
    """
    # Params: (kappa, m12) - with m21 = 0.0
    kappa, m12 = params_model(params)
    m21 = 0.0
    tau = 10.0  # time in the past of split

    if FIXED == 'kappa':
        m12 = m12[0]

    # Define the grid we'll use
    grid = dadi.Numerics.default_grid(pts)

    # Define the phi_ancestral, i.e. phi for the equilibrium ancestral population
    phi_ancestral = dadi.PhiManip.phi_1D(grid)

    # Split the ancestral population into two population
    phi = dadi.PhiManip.phi_1D_to_2D(grid, phi_ancestral)

    # Define the sudden decline event at a time tau in past

    phi = dadi.Integration.two_pops(phi, grid, tau, nu1=1.0, nu2=1.0*kappa, m12=m12, m21=m21)

    # Remove population 2 from phi
    phi = dadi.PhiManip.remove_pop(phi, grid, 2)

    # Calculate the spectrum from phi
    sfs = dadi.Spectrum.from_phi(phi, ns, (grid,))

    return sfs


def parameters_optimization(p0, sfs, model_func, pts_list, lower_bound, upper_bound,
                            verbose=0):
    """
    Parameters optimization.

    The upper_bound and lower_bound lists are use in optimization. Occasionally the optimizer
    will try wacky parameters values. We in particular want to exclude values with very long
    times, very small population sizes, or very high migration rates, as they will take a long
    time to evaluate.
    Parameters can be (kappa), (tau), (kappa, tau), etc.

    Parameter
    ---------
    p0: list
        Initial parameters - this is our initial guess, which is somewhat arbitrary.
    sfs
        Spectrum with data
    model_func
        Function to evaluate model spectrum - extrapolated.
    pts_list: list
        the grid point use for extrapolation
    lower_bound: list
        Lower bound on parameter values. If not None, must be of same length as p0
    upper_bound: list
        Upper bound on parameter values. If not None, must be of same length as p0

    Return
    ------
    popt: list
        Optimize log(params) to fit model to data using the BFGS method.
    """
    # Perturb our parameters before optimization. This does so by taking each parameter a up
    # to a factor of two up or down.
    p0 = dadi.Misc.perturb_params(p0, fold=1, upper_bound=upper_bound, lower_bound=lower_bound)

    # Do the optimization. By default we assume that theta is a free parameter, since it's
    # trivial to find given the other parameters. If you want to fix theta, add a
    # multinom=False to the call.
    # The maxiter argument restricts how long the optimizer will run. For real runs, you will
    # want to set this value higher (at least 10), to encourage better convergence. You will
    # also want to run optimization several times using multiple sets of intial parameters, to
    # be confident you've actually found the true maximum likelihood parameters.

    if verbose:
        print('Beginning optimization ************************************************')

    popt = dadi.Inference.optimize_log(p0, sfs, model_func, pts_list,
                                       lower_bound=lower_bound, upper_bound=upper_bound,
                                       verbose=verbose, maxiter=1)

    if verbose:
        # The verbose argument controls how often progress of the optimizer should be
        # printed. It's useful to keep track of optimization process.
        print('Finished optimization **************************************************')
        print('Best-fit parameters: {}'.format(popt))

    return popt


def inference(pts_list, model_func, fixed=None, value=None, verbose=0, path="./Data/",
              name="SFS"):
    """
    Dadi inference.

    Parameter
    ---------
    pts_list: list
        the grid point use for extrapolation
    model_func: function
        the custom model_func

    Return
    ------
    ll_model: float
        likelihood of the data
    theta: float
        the optimal value of theta given the model
    model: list
        the sfs inferred
    """
    global FIXED, VALUE
    FIXED = fixed
    VALUE = value

    # Load the data
    observed_sfs = dadi.Spectrum.from_file("{}{}.fs".format(path, name))
    ns = observed_sfs.sample_sizes

    # Make the extrapolation version of our demographic model function
    model_func_extrapolated = dadi.Numerics.make_extrap_log_func(model_func)

    # Optimisation of model parameters
    if model_func.__name__ == 'constant_model':
        inferred_sfs = model_func_extrapolated(ns, pts_list)

    else:
        # Set up:
        #   - p0: initial guess for the parameters, which is somewhat arbitrary
        #   - lower & upper bound for the optimization

        if FIXED == 'tau':  # Fixed param: (Tau) & Param evaluates: (Kappa)
            # Param: (Kappa)
            p0, lower_bound, upper_bound = [1.0], [1e-4], [1e3]

        elif FIXED == 'kappa':  # Fixed param: (Kappa) & Param evaluates: (Tau)
            # Param: (Tau)
            p0, lower_bound, upper_bound = [0.], [1e-4], [1e4]

        elif FIXED == 'm12':  # Fixed param: (m12), the migration rate into 1 from 2
            # Pram: (m12)
            p0, lower_bound, upper_bound = [1.0], [1e-4], [1e3]

        else:  # Params evaluate: (Kappa, Tau) or (Kappa, m12)
            p0, lower_bound, upper_bound = [1.0, 0.], [1e-4, 1e-4], [1e3, 1e4]

        popt = parameters_optimization(p0, observed_sfs, model_func_extrapolated, pts_list,
                                       lower_bound, upper_bound, verbose=verbose)
        
        # Simulated frequency spectrum
        inferred_sfs = model_func_extrapolated(popt, ns, pts_list)

        # Keep track of parameters after optimization
        if FIXED == 'tau':
            params_estimated = {'Tau': VALUE, 'Kappa': popt[0]}
        elif FIXED == 'kappa':
            params_estimated = {'Tau': popt[0], 'Kappa': VALUE}
        elif FIXED == 'm12':
            params_estimated = {'Tau': popt[0], 'Kappa': VALUE}
        elif model_func.__name__ == 'sudden_decline_model':
            params_estimated = {'Tau': popt[1], 'Kappa': popt[0]}
        else:
            params_estimated = {'m12': popt[1], 'Kappa': popt[0]}

    # Log-likelihood of the data (sfs) given the model
    ll_model = dadi.Inference.ll_multinom(inferred_sfs, observed_sfs)

    # The optimal value of theta given the model
    theta = dadi.Inference.optimal_sfs_scaling(inferred_sfs, observed_sfs)

    # Remove 0/n & n/n
    inferred_sfs = list(inferred_sfs)[1:-1]

    # Return
    if model_func.__name__ == 'constant_model':
        return ll_model, inferred_sfs
    
    params_estimated['Theta'] = theta
    return ll_model, inferred_sfs, params_estimated


if __name__ == "__main__":
    sys.exit()  # No actions desired
