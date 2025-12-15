#!/bin/bash
#SBATCH --job-name=agnews_mlp_eval
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/agnews_mlp_eval.out
#SBATCH --error=logs/agnews_mlp_eval.err

module purge
source /scratch/hz3916/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/hz3916/miniconda3/envs/textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

python eval_step4_agnews.py \
  --data_root scripts/agnews_emb \
  --seeds 0 1 2 3 4 \
  --paths \
    agnews_mlp_fullbptt_ipc01_s0.h5 \
    agnews_mlp_fullbptt_ipc10_s1.h5 \
    agnews_mlp_fullbptt_ipc50_s2.h5 \
    agnews_mlp_ratbptt_ipc01_s0.h5 \
    agnews_mlp_ratbptt_ipc10_s1.h5 \
    agnews_mlp_ratbptt_ipc50_s2.h5 \
    agnews_mlp_ratbptt_boost_ipc01_s0.h5 \
    agnews_mlp_ratbptt_boost_ipc05_s1.h5 \
    agnews_mlp_ratbptt_boost_ipc10_s2.h5 \
    agnews_mlp_ratbptt_boost_ipc15_s3.h5 \
    agnews_mlp_ratbptt_boost_ipc20_s4.h5 \
    agnews_mlp_ratbptt_boost_ipc50_s5.h5
