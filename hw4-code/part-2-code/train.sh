#!/bin/bash
#SBATCH --partition=a100_dev
#SBATCH --job-name=t5_train
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --output=logs/t5_train_%j.out
#SBATCH --error=logs/t5_train_%j.err
#SBATCH --cpus-per-task=4


source ~/.bashrc
conda activate hw4-part-2-nlp
nvidia-smi

mkdir -p logs checkpoints 

python train_t5.py \
    --finetune \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --max_n_epochs 30 \
    --patience_epochs 5 \
    --num_warmup_epochs 3 \
    --scheduler_type cosine \
    --batch_size 8 \
    --test_batch_size 16 \
    --experiment_name baseline_no_prefix_greedy_1


