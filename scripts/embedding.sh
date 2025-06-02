#!/bin/bash
#
#---------------------------------------------
# SLURM job script for single CPU/GPU
#---------------------------------------------
#
#
#SBATCH --job-name=ants_emb              # Job name
#SBATCH --output=./logs/ants_emb_%j.log  # Output file
#SBATCH --time=24:00:00                  # Maximum walltime
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --mem=100G                       # Memory allocation
#SBATCH --partition=gpu100               # Partition to use (e.g., gpu)
#SBATCH --gres=gpu:1                     # Number of GPUs requested (adjust as needed)
#
# Load your environment or module
##source ~/.bashrc                      # Ensure you load your bash profile
module load conda
conda activate crl                      # Activate your conda environment

# general variables 
data_dir="data/v2"
environment="supervised"
background=True


# Run experiments
srun python scripts/run_emb.py --data_dir $data_dir --environment $environment #--background $background 