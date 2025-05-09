#!/bin/bash
#SBATCH --mem=64GB
#SBATCH --time=2:00:00
#SBATCH --job-name=data
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

source linesegmentation/sea/bin/activate
srun python noise_designer.py
