"""
This module allows the generation of different demographic scenarios from simulator msprime.

Various scenarios are set up:
  -       Constant model: control scenario
  - Sudden decline model: decline of force kappa at a time tau
  -  Sudden growth model: growth of force kappa at a time tau
"""

import sys
import numpy as np
import msprime


def msprime_debugger(configuration_pop, history, migration_matrix):
    debugger = msprime.DemographyDebugger(
        population_configurations=configuration_pop, demographic_events=history,
        migration_matrix=migration_matrix
    )
    debugger.print_history()


def constant_model(params, debug):
    """
    Constant model, i.e. population size is constant - control scenario.

    Parameter
    ---------
    params: dictionary

    Return
    ------
    configuration_pop: list
        the configuration of the constant population - size, growth.
    history: None
        constant model so no history
    """
    sample, pop = params['sample_size'], params['Ne']

    configuration_pop = [
        msprime.PopulationConfiguration(sample_size=sample, initial_size=pop, growth_rate=0)
    ]
    history, migration_matrix = None, None

    if debug:
        msprime_debugger(configuration_pop, history, migration_matrix)

    return configuration_pop, history, migration_matrix


def sudden_decline_model(params, debug):
    """
    Sudden decline model of the population (force kappa) at a time tau in the past.

    Parameter
    ---------
    params: dictionary
      - Tau: the lenght of time ago at which the event (decline, growth) occured
      - Kappa: the growth or decline force

    Return
    ------
    configuration_pop: list
        the initial configuration of the population, i.e. at time 0
    history: list
        the observed demographic change in the population at tau time - decline of force kappa
    """
    sample, pop, tau, kappa = \
        params['sample_size'], params['Ne'], params['Tau'], params['Kappa']

    configuration_pop = [
        msprime.PopulationConfiguration(sample_size=sample, initial_size=pop, growth_rate=0)
    ]

    # Ancestral population
    history = [
        msprime.PopulationParametersChange(time=tau, initial_size=pop*kappa, growth_rate=0)
    ]
    migration_matrix = None

    if debug:
        msprime_debugger(configuration_pop, history, migration_matrix)

    return configuration_pop, history, migration_matrix


def sudden_growth_model(params, debug):
    """
    Sudden growth model of the population (force kappa) at a time tau in the past.

    Parameter
    ---------
    params: dictionary
      - Tau: the lenght of time ago at which the event (decline, growth) occured
      - Kappa: the growth or decline force

    Return
    ------
    configuration_pop: list
        the initial configuration of the population, i.e. at time 0
    history: list
        the observed demographic change in the population at tau time - growth of force kappa
    """
    sample, pop, tau, kappa = \
        params['sample_size'], params['Ne'], params['Tau'], params['Kappa']

    configuration_pop = [
        msprime.PopulationConfiguration(sample_size=sample, initial_size=pop, growth_rate=0)
    ]
    history = [
        msprime.PopulationParametersChange(time=tau, initial_size=pop/kappa, growth_rate=0)
    ]
    migration_matrix = None

    if debug:
        msprime_debugger(configuration_pop, history, migration_matrix)

    return configuration_pop, history, migration_matrix


def twopops_migration_model(params, debug):
    """
    Migration model with msprime.

    Populations:
      - Population 1 of size n1 - we only choose samples from this population
      - Population 2 of size n2 with n2 = kappa * n1.

    Migration
      - There is no migration from population 1 to 2
      - There is some migrations from population 2 to 1

    Parameter
    ---------
    params: dictionary
      - Kappa: ratio of population's 1 size to population's 2 size
      - m12: migration rate into population 1 from 2
      - m21: migration rate into population 2 from 1 - by default it's 0
    """
    sample, pop, kappa, m12, m21 = \
        params['sample_size'], params['Ne'], params['Kappa'], params['m12'], \
        params['m21']

    # The list of PopulationConfiguration instances describing the sampling configuration, the
    # relative sizes and growth rates of the population to be simulated.
    configuration_pop = [
        msprime.PopulationConfiguration(sample_size=sample, initial_size=pop, growth_rate=0),
        msprime.PopulationConfiguration(sample_size=0, initial_size=kappa*pop, growth_rate=0)
    ]
    history = None

    # The matrix describing the rates of migration between all pairs of populations.
    # It's an N*N matrix with N the number of populations defined in configuration_pop.
    # Each element of the matrix Mj,k defines the fraction of population j that consists of
    # migrants from population k in each generation.
    migration_matrix = [
        [0, m12],
        [m21, 0]
    ]

    if debug:
        msprime_debugger(configuration_pop, history, migration_matrix)

    return configuration_pop, history, migration_matrix


def msprime_simulation(model, params, debug=False):
    """
    Population simulation with msprime (msprime 0;x).

    IMPORTANT
    This function is deprecated (but supported indefinitely); msprime_simulate_variants use
    sim_ancestry() of msprime 1.x.

    Parameter
    ---------
    model: function
        (constant, sudden declin, sudden growth, etc.)
    params: dictionary
        - sample_size: the number of sampled monoploid genomes
        - Ne: the effective (diploid) population size
        - rcb_rate: the rate of recombinaison per base per generation
        - mu: the rate of infinite sites mutations per unit of sequence length per generation
        - length: the length of the simulated region in bases
    tau: the lenght of time ago at which the event (decline, growth) occured
    kappa: the growth or decline force
    debug: Boolean
        1: print msprime debugger, 0: nothing

    Some notes about the simulation of ancestry with the method simulate of Msprime 0.x:
        - sample_size
          The sample_size is the number of monoploid genomes.
          So if sample_size is set to 20 that means there are 2N=20 genomes sampled

    Return
    ------
    sfs: list
        Site frequency Spectrum (sfs) - allele mutation frequency
    variants: list
        List of position and genotypes for each variant with 0 the ancestral state and 1 the
        alternative one.
    """
    demography = model(params, debug)

    tree_seq = msprime.simulate(
        length=params["length"], recombination_rate=params["rcb_rate"],
        mutation_rate=params["mu"], population_configurations=demography[0],
        demographic_events=demography[1], migration_matrix=demography[2]
    )

    if debug:
        print(tree_seq.first().draw(format="unicode"))

    sfs = [0] * (params["sample_size"] - 1)
    for variant in tree_seq.variants():
        # Generate SFS
        _, counts = np.unique(variant.genotypes, return_counts=True)
        freq_mutation = counts[1]-1
        sfs[freq_mutation] += 1

    return sfs  #, variants


def msprime_simulate_variants(params, debug=False):
    """
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
        - rcb_rate: the rate of recombinaison per base per generation
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
        If False that means mutations are placed at continuous, float coordinates (msprime 0.x)

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
    demography = msprime.Demography()

    # Population actuelle au temps 0
    demography.add_population(initial_size=params['Ne'], growth_rate=0.)

    # Ancestral population
    demography.add_population_parameters_change(
        time=params['Tau'], population=0, initial_size=params['Ne']*params['Kappa'],
        growth_rate=0.)

    if debug:
        print(demography.debug())

    # Simulation of ancestry
    ts = msprime.sim_ancestry(
        samples=int(params['sample_size'] / 2), demography=demography, ploidy=2,
        sequence_length=params['length'], discrete_genome=True,
        recombination_rate=params['rcb_rate']
    )

    # Mutation model to use - binary mutation model
    #   - Allele ["0", "1"]
    #   - Root distribution [1., 0.], i.e. all ancestral states will be 0
    mutation_model = msprime.BinaryMutationModel(state_independent=False)

    # Genetic variation of the data with mutation
    mts = msprime.sim_mutations(tree_sequence=ts, rate=params['mu'], model=mutation_model)

    sfs, variants = [0] * (params["sample_size"] - 1), []
    for variant in mts.variants():
        _, counts = np.unique(variant.genotypes, return_counts=True)

        if len(counts) != 1:
            # SFS
            freq_mutation = counts[1]
            sfs[freq_mutation-1] += 1

            # Genotype
            variants.append((variant.site.position, variant.genotypes))
            
        # QUESTION: some variants with [0 0 ... 0 0] or [1 1 ... 1 1], je ne comprends pas ?
        # print(variant.site.position, variant.alleles, variant.genotypes)

    return sfs, variants


if __name__ == "__main__":
    sys.exit()  # No actions desired
