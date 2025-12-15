#!/bin/bash
#SBATCH --job-name=agnews_eval_syn_s2-5
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1   
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/step4_eval_syn_s2-5.out
#SBATCH --error=logs/step4_eval_syn_s2-5.err

source /scratch/hz3916/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/hz3916/miniconda3/envs/textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

python eval_step4_agnews.py \
  --data_root scripts/agnews_emb \
  --seeds 0 1 2 3 4 \
  --paths out_step3_agnews_mlp_ipc20_s2.h5 out_step3_agnews_mlp_ipc25_s3.h5 out_step3_agnews_mlp_ipc30_s4.h5 out_step3_agnews_mlp_ipc35_s5.h5
