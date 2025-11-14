#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err

module purge
source ~/.bashrc
conda activate hw4-part-1-nlp

nvidia-smi

python3 main.py --eval --model_dir out_augmented



