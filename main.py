import linckage as lk
import plot as plt
import argparse
import sys
import pandas as pd
import numpy as np

def get_arguments():
    """Retrieves the arguments of the program.
      Returns: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, usage=
                                     "{0} -h"
                                     .format(sys.argv[0]))
    parser.add_argument('-t', dest='type', type=str, help="Fastq file")
    parser.add_argument('-o', dest='output_file', type=str,
                        default=None,
                        help="Output contigs in fasta file")
    parser.add_argument('-ht', dest='output_ht', type=str,
                        default=None,
                        help="Output contigs in fasta file")
    parser.add_argument('-d', dest='data_ht', type=str,
                        default=None,
                        help="Output contigs in fasta file")
    return parser.parse_args()


def main():
    d_type = {"sfs": sfs, "ld": ld}
    args = get_arguments()
    if args.output_ht is not None:
        data = pd.read_csv(args.data_ht)
        plt.plot_heatmap(data=data, title="", cbar="", filout=args.output_ht)
    else:
        params = {"sample_size": 10, "Ne": 1, "ro": 8e-3, "mu": 8e-3,  "Tau": 1.0,
        "Kappa": 1.0 , "length": int(1e5), "type": "ld"}
        data_ld = lk.senario(d_type[args.type], params)
        plt.plot_dist(args.type, data_ld,
        "scenario_{}".format(args.type), True)
        kappa_range = np.exp(np.arange(-3.5, 2.8, 0.1))
        tau_range = np.exp(np.arange(-4, 2.3, 0.1))
        data  = lk.data_heat_map(d_type[args.type], kappa_range, tau_range, params)
        data.to_csv(args.output_file, index=False)
    # pkl.dump(generat_senar(params), "out")
     # params = {"sample_size":10, "Ne": 1, "ro": 8e-2, "mu": 8e-3,  "Tau": 1.0, "length": int(1e5)}
     # LD_senario_ro(params, "ro")
     # LD_mu_senario(params, "mu")

     # print(data)


if __name__ == "__main__":
    main()
