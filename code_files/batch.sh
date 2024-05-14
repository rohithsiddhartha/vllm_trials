#!/bin/bash
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH -t 8:00:00               # Max runtime of 8 hours
#SBATCH --partition=gpu-preempt  # Use the GPU preempt partition
#SBATCH --constraint=a100        # Use NVIDIA A100 GPU
#SBATCH --mem=30G                # Request 30GB of memory
#SBATCH --job-name=d3_storygen_olmo
#SBATCH --output=d3_storygen_olmo.out
#SBATCH --error=d3_storygen_olmo.err
#SBATCH --mail-user=rbheemreddy@umass.edu
#SBATCH --mail-type=ALL

# Load Miniconda
module load miniconda/22.11.1-1
# Activate your Conda environment
conda activate vllm

# Navigate to the directory containing your Python script
cd /home/rbheemreddy_umass_edu/

# Execute your Python script
# python direction3/storygen.py

python3 vllm_trials/code_files/storygen.py
# --folder_path "vllm_trials/Expansion/direction2" --system_prompt_file "system_prompt.txt"  --direction "d2" --api_key_file "api_key.txt" --model "gpt-4"

