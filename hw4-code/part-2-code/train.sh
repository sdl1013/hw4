#!/bin/bash
#SBATCH --job-name=t5_train
#SBATCH --partition=v100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/t5_train_%j.out
#SBATCH --error=logs/t5_train_%j.err
#SBATCH --cpus-per-task=4


module purge
source ~/.bashrc
conda activate hw4-part-2-nlp
nvidia-smi

mkdir -p logs checkpoints 

python train_t5.py \
    --finetune \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --max_n_epochs 10 \
    --patience_epochs 3 \
    --num_warmup_epochs 1 \
    --scheduler_type cosine \
    --batch_size 8 \
    --test_batch_size 8 \
    --experiment_name baseline_add_prefix_3


