#!/bin/bash
#
#---------------------------------------------
# SLURM job script for single CPU/GPU
#---------------------------------------------
#
#
#SBATCH --job-name=ants_gen             # Job name
#SBATCH --output=./logs/ants_gen%j.log  # Output file
#SBATCH --time=08:00:00                 # Maximum walltime
#SBATCH --ntasks=2                      # Number of tasks
#SBATCH --mem=100G                      # Memory allocation
#SBATCH --partition=gpu                 # Partition to use (e.g., gpu)
#SBATCH --gres=gpu:1                    # Number of GPUs requested (adjust as needed)
#
# Load your environment or module
##source ~/.bashrc                      # Ensure you load your bash profile
module load conda
conda activate crl                      # Activate your conda environment

# general variables 
ref_preprocessed="preprocessed/0.99"
tar_preprocessed="preprocessed/0.85"
# ref_preprocessed="original"
# tar_preprocessed="original"
sc="all"
task="or"


# Run experiments
srun python scripts/run_gen.py --sc $sc --task $task --ref_preprocessed $ref_preprocessed --tar_preprocessed $tar_preprocessed 