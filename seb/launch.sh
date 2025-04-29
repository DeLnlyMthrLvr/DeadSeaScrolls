#!/bin/bash
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --job-name=dss
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --nodes=1

module --force purge
cd dlp/seb/
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1 
source ../.venv/bin/activate
srun python3 train.py