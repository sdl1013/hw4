#!/bin/bash
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16GB
#SBATCH --time=01:00:00
#SBATCH --output=%j.out
#SBATCH --error=%j.err



source ~/.bashrc
conda activate hw4-part-2-nlp
nvidia-smi

python compute_stats.py 



