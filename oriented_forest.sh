
#!/bin/bash
# Nom du job
# -N msprime
# Short pour un job < 12h
#$ -q short.q
# Adresse à envoyer
# -M abdelmajid.omarjee@college-de-france.fr
# Envoie mail - (b)egin, (e)nd, (a)bort & (s)uspend
# -m as
# Sortie standard
#$ -o $HOME/work/Out2
# Sortie d'erreur
#$ -e $HOME/work/Err2
#$ -pe thread 10
conda activate sei-3.8.5
python /home/aomarjee/work/demogr/oriented_forest.py
python /home/aomarjee/work/demogr/oriented_forest_2.py
conda deactivate
