#!/bin/bash
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --job-name=lstrain
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=v100:1
#SBATCH --nodes=1

module purge
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

source ../linesegmentation/sea/bin/activate
srun python train.py
