#!/bin/bash
#
#---------------------------------------------
# SLURM job script for single CPU/GPU
#---------------------------------------------
#
#
#SBATCH --job-name=ants_main             # Job name
#SBATCH --output=./logs/ants_main%j.log  # Output file
#SBATCH --time=04:00:00                  # Maximum walltime
#SBATCH --ntasks=1                       # Number of tasks
#SBATCH --mem=10G                        # Memory allocation
#SBATCH --partition=gpu100              # Partition to use (e.g., gpu)
#SBATCH --gres=gpu:1                    # Number of GPUs requested (adjust as needed)
#
# Load your environment or module
##source ~/.bashrc                      # Ensure you load your bash profile
module load conda
conda activate crl                      # Activate your conda environment

# general variables 
dir="./data/istant_lq"
preprocessed="preprocessed/0.99"
#preprocessed="original"

# Run experiments
srun python scripts/run_main.py --data_dir $dir --results_dir $dir --preprocessed $preprocessed