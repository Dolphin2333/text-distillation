#!/bin/bash
#SBATCH --job-name=agnews_eval_fullbptt_5_20
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_mlp_fullbptt_ipc5_20.out
#SBATCH --error=logs/eval_mlp_fullbptt_ipc5_20.err

source /scratch/hz3916/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/hz3916/miniconda3/envs/textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

python eval_step4_agnews.py \
  --data_root scripts/agnews_emb \
  --seeds 0 1 2 3 4 \
  --paths \
    checkpoints/agnews_mlp_fullbptt_ipc05_s0.h5 \
    checkpoints/agnews_mlp_fullbptt_ipc20_s1.h5

echo "Evaluation jobs for Full-BPTT IPC 5/20 submitted."
