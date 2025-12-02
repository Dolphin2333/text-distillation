#!/bin/bash
#SBATCH --job-name=agnews_eval_syn
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/step4_eval_syn.out
#SBATCH --error=logs/step4_eval_syn.err

source ~/.bashrc
conda activate textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

python eval_step4_agnews.py \
  --data_root scripts/agnews_emb \
  --seeds 0 1 2 3 4 \
  --paths out_step3_agnews_mlp_ipc10_s0.h5 out_step3_agnews_mlp_ipc15_s1.h5
