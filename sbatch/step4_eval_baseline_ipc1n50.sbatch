#!/bin/bash
#SBATCH --job-name=agnews_eval_base_ipc1n50
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1   
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/step4_eval_base_ipc1n50.out
#SBATCH --error=logs/step4_eval_base_ipc1n50.err

source /scratch/hz3916/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/hz3916/miniconda3/envs/textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

python eval_baseline_random_agnews.py \
  --data_root scripts/agnews_emb \
  --ipcs 1 50 \
  --seeds 0 1 2 3 4
