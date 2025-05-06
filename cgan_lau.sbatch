#!/bin/bash
#SBATCH --job-name=cyclegan
#SBATCH --output=runs/%j/cyclegan_log.out
#SBATCH --error=runs/%j/cyclega_err.err
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --time=4:00:00

module load CUDA/12.4.0

source /home1/s6133800/DeadSeaScrolls/.scroll/bin/activate

python3 cyclegan_train.py