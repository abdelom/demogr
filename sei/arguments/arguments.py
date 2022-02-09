"""
Define command-line options, arguments and sub-commands by using argparse.
"""

import argparse
import sys


def data_type(value):
    try:
        value = int(value)
    except ValueError as type_error:
        raise argparse.ArgumentTypeError('Value must be an integer !') from type_error

    if value < 1:
        raise argparse.ArgumentTypeError('Value must be an integer >= 1')

    return value


def arguments():
    """
    Define arguments.
    """
    parser = argparse.ArgumentParser()

    # Define the subparser
    subparsers = parser.add_subparsers(dest='analyse', required=True)

    #############################################
    # Generate various set of data with msprime #
    #############################################
    data = subparsers.add_parser('data', help="Generate various unfolded sfs with msprime")
    data.add_argument(
        '--model', dest='model', required=True, choices=['decline', 'migration', 'cst'],
        help="Kind of scenario to use for the generation of sfs with msprime - decline"
        ", migration, cst for constant population"
    )

    group = \
        data.add_mutually_exclusive_group(required=(len(sys.argv) > 3 and sys.argv[3] != 'cst'))

    group.add_argument(
        '--job', dest='job', type=data_type,
        help="Simulation with msprime for a given tau & kappa - to sumit job-array with migale"
        " cluster, from 1 to 4225"
    )

    group.add_argument(
        '--file', dest='file', action='store_true',
        help="Determine the length factor for each (tau, kappa) pairs"
    )

    data.add_argument(
        '--typ', dest='typ', required=True, choices=['sfs', 'vcf'], default='sfs',
        help="Generate a set of SFS for Dadi & Stairway or a set of VCF for SMC++"
    )

    #############################################
    # Msprime verification                      #
    #############################################
    msprime = subparsers.add_parser(
        'msprime', help="Check unfolded sfs generated with msprime for various scenarios"
    )

    #############################################
    # Optimisation Dadi (old)                   #
    #############################################
    opt = subparsers.add_parser('opt', help="Compute optimisation of dadi's parameters")
    opt.add_argument(
        '--nb', dest='number', type=int, required=True,
        help="""Determine for a given number of sampled genomes n, the error rate of the
        inference of 100 observed for various mutation rate mu"""
    )

    #############################################
    # Optimisation SMC                          #
    #############################################
    optsmc = subparsers.add_parser('optsmc', help="Compute optimization for smc++")
    optsmc.add_argument(
        '--model', dest="model", choices=['decline', 'growth', 'cst'], required=True,
        help="Perform the optimization for a sudden decline (Tau=0., Kappa=1.), sudden growth "
        "(Tau=0., Kappa=-1.) or constant (Tau=0., Kappa=0.) model"
    )
    optsmc.add_argument(
        '--job', dest='job', type=data_type, required=True, help="file to analyse"
    )

    #############################################
    # Optimisation SNPs                         #
    #############################################
    optsnp = subparsers.add_parser('optsnp', help="Compute optimization for smc++")
    optsnp.add_argument(
        '--job', dest='job', type=data_type, required=True, help="file to analyse"
    )

    #############################################
    # Optimisation SNPs for Dadi and msprime    #
    #############################################
    optdadi = subparsers.add_parser('optdadi', help="Compute optimization for dadi & msprime")
    optdadi.add_argument(
        '--model', dest="model", choices=['decline', 'growth', 'cst'], required=True,
        help="Perform the optimization for a sudden decline (Tau=0., Kappa=1.), sudden growth "
        "(Tau=0., Kappa=-1.) or constant (Tau=0., Kappa=0.) model"
    )
    optdadi.add_argument(
        '--job', dest='job', type=data_type, required=True, help="file to analyse"
    )
    
    #############################################
    # Inference of demographic history          #
    #############################################
    inf = subparsers.add_parser('inf', help="Compute inference of demographic history")
    tool = inf.add_mutually_exclusive_group(required=True)

    # Dadi
    tool.add_argument('-dadi', action='store_true',
                      help="Inference of demographic history with Dadi")
    inf.add_argument(
        '--param', dest="param", choices=['tau', 'kappa', 'm12'], default=None,
        help="Fixed parameters, either (tau), (kappa) or (m12), m12 is the migration rate from"
        " population 2 to 1."
    )
    inf.add_argument(
        '--value', dest='value', type=float, default=None,
        required='--param' in sys.argv is not None,
        help="Value of the fixed parameters for the inference - value in log scale"
    )

    # Stairway
    tool.add_argument('-stairway', action='store_true',
                      help="Inference of demographic history with Stairway plot 2")

    # SMC++
    tool.add_argument('-smc', action='store_true',
                      help="Inference of demographic history with SMC++")

    # Required argument for booth Dadi, stairway & smc
    inf.add_argument(
        '--model', dest="model", choices=['decline', 'migration', 'cst'], required=True,
        help="Simulation model used for the inference, in the case of dadi also indicate the "
        "population model for the inference"
    )
    inf.add_argument(
        '--job', dest='job', type=data_type, required=True,
        help="Inference with dadi for a given tau/m12 & kappa - to sumit job-array with migale"
        " cluster, if param empty from 1 to 4225 else from 1 to 65"
    )

    # Optional argument
    inf.add_argument(
        '--fold', dest='fold', action="store_true", help="To work with folded SFS"
    )


    #############################################
    # Plot error rate                           #
    #############################################
    er = subparsers.add_parser('er', help="Plot error rate of simulation with dadi")

    #############################################
    # Assessment of inferences                  #
    #############################################
    ases = subparsers.add_parser('ases', help="Evaluation of inference")
    # ases.add_argument(
    #     '--tool', dest='tool', required=True,
    #     help"Tools to evaluate - dadi, stairway plot, etc."
    # )
    ases.add_argument(
        '--param', dest='param', choices=['tau', 'kappa', 'tau-kappa'], required=True,
        help="Parameter to evaluate - (tau) - default, (kappa), (kappa, tau)"
    )

    # Inference with stairway plot
    stairway = subparsers.add_parser('stairway', help="Stairway plots")

    return parser.parse_args()
