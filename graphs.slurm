#!/bin/bash
#SBATCH --job-name=graphs   # nom du job
#SBATCH --output=graphs.out  # fichier de sortie (%j = job ID)
#SBATCH --error=graphs.err   # fichier d'erreur (%j = job ID)
#SBATCH --constraint=h100               # demander des GPU H100 80 Go
#SBATCH --nodes=1                       # reserver 1 noeud
#SBATCH --ntasks=1                      # 1 tache shell (les shards sont des sous-processus)
#SBATCH --gres=gpu:1                    # reserver 1 GPU
#SBATCH --cpus-per-task=8               # reserver 8 CPU
#SBATCH --time=20:00:00                 # temps maximal d'allocation "(HH:MM:SS)"
#SBATCH --hint=nomultithread            # desactiver l'hyperthreading
#SBATCH --account=ehe@h100              # comptabilite H100

module purge
module load arch/h100
module load pytorch-gpu/py3/2.4.0

source /lustre/fsn1/projects/rech/ehe/udq27fb/mil_survival/pgcn_venv/bin/activate
# TRIDENT local en premier pour override la version pip (contient Maintenance, IO custom, etc.)
export PYTHONPATH=/lustre/fsn1/projects/rech/ehe/udq27fb/mil_survival/pgcn_venv/lib/python3.11/site-packages:$PYTHONPATH

set -x

python /lustre/fsn1/projects/rech/ehe/udq27fb/mil_survival/create_graphs_from_csv.py
