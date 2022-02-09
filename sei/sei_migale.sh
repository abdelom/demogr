#!/bin/bash

# Nom du job
# -N msprime

# Number of separate submissions to the cluster
#$ -t 1-4225

# Short pour un job < 12h
#$ -q long.q

# Adresse à envoyer
# -M pierre.imbert@college-de-france.fr

# Envoie mail - (b)egin, (e)nd, (a)bort & (s)uspend
# -m as

# Sortie standard
#$ -o $HOME/work/Out

# Sortie d'erreur
#$ -e $HOME/work/Err
 
conda activate sei-3.8.5
python /home/pimbert/work/linckage.py

conda deactivate
