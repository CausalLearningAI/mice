#!/bin/bash
#
#---------------------------------------------
# SLURM job script for single CPU/GPU
#---------------------------------------------
#
#
#SBATCH --job-name=ants_track            # Job name
#SBATCH --output=logs/ants_track_%j.log  # Output file
#SBATCH --time=8:00:00                   # Maximum walltime
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --mem=10G                        # Memory allocation
##SBATCH --partition=gpu                  # Partition to use (e.g., gpu)
#SBATCH --gres=gpu:0                     # Number of GPUs requested (adjust as needed)
#
# Load your environment or module
##source ~/.bashrc                      # Ensure you load your bash profile
module load conda
conda activate crl                      # Activate your conda environment

# Run experiments
srun python scripts/run_track.py 