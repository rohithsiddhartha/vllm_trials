#!/bin/bash
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH -t 8:00:00               # Max runtime of 6 hours
#SBATCH --partition=gpu-preempt  # Use the GPU preempt partition
#SBATCH --constraint=2080ti        # Use NVIDIA A100 GPU
#SBATCH --mem=11G                # Request 11GB of memory
#SBATCH --job-name=d2_cs_eval
#SBATCH --output=d2_cs_storygen.out
#SBATCH --error=d2_cs_storygen.err
#SBATCH --mail-user=rbheemreddy@umass.edu,aatmakuru@umass.edu
#SBATCH --mail-type=ALL

# Load Miniconda
module load miniconda/22.11.1-1
# Activate your Conda environment
conda activate vllm

# Navigate to the directory containing your Python script
cd /home/rbheemreddy_umass_edu/vllm_trials

# Execute your Python script
# python direction3/storygen.py

bash code_files/cs.sh
