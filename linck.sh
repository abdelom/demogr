#!/bin/bash
# Nom du job
# -N msprime
# Short pour un job < 12h
#$ -q long.q
# Adresse à envoyer
# -M abdelmajid.omarjee@college-de-france.fr
# Envoie mail - (b)egin, (e)nd, (a)bort & (s)uspend
# -m as
# Sortie standard
#$ -o $HOME/work/linck.out
# Sortie d'erreur
#$ -e $HOME/work/linck.err
conda activate sei-3.8.5
python /home/aomarjee/work/demogr/main.py -t sfs -o /home/aomarjee/work/sfs
python /home/aomarjee/work/demogr/main.py -t ld -o /home/aomarjee/work/ld
conda deactivate
