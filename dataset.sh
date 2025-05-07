#!/bin/bash
#SBATCH --mem=GB
#SBATCH --time=14:00:00
#SBATCH --job-name=dataset
#SBATCH --partition=regular
#SBATCH --nodes=1

module purge
cd art_restoration/
module load PyTorch-bundle/2.1.2-foss-2023a-CUDA-12.1.1

source art/bin/activate
srun python train.py --fragments 2 --ls 128 --checkpoint 20241217-080842__gen_8M_residual_actually_big_128__7SvGaQ
