#!/bin/bash
#SBATCH --partition=all
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=72:00:00
#SBATCH --mem=30G
#SBATCH --job-name=n_beta
#SBATCH --output=n_beta_%j.log

source /home/ukmot/miniconda3/etc/profile.d/conda.sh
conda activate qec_env

python3 plot_n_beta_curves.py
