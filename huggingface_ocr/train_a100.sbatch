#!/bin/bash
#SBATCH --job-name=ocr_training
#SBATCH --output=logs/ocr_training_%j.out
#SBATCH --error=logs/ocr_training_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --nodes=1
#SBATCH --time=2:00:00

# Load necessary modules (optional, depends on your cluster)
module load CUDA/12.4.0

# Activate your environment (if using conda or venv)
source /home3/s3799042/lc0_venv2/bin/activate

# Run your training script
python3 huggingface_ocr.py
