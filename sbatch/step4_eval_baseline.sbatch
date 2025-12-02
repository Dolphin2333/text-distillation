#!/bin/bash
#SBATCH --job-name=agnews_eval_base
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/step4_eval_base.out
#SBATCH --error=logs/step4_eval_base.err

source ~/.bashrc
conda activate textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

python eval_baseline_random_agnews.py \
  --data_root scripts/agnews_emb \
  --ipcs 5 10 15 20 \
  --seeds 0 1 2 3 4
