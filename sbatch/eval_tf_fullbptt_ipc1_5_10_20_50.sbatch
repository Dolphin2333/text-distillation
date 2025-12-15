#!/bin/bash
#SBATCH --job-name=agnews_eval_tf_fullbptt
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval_tf_fullbptt.out
#SBATCH --error=logs/eval_tf_fullbptt.err

module purge
source /scratch/hz3916/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/hz3916/miniconda3/envs/textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

python eval_step5_distill.py \
  --data_root scripts/agnews_emb \
  --val_emb_file agnews_val_emb.pt \
  --val_label_file agnews_val_labels.pt \
  --paths \
    checkpoints/out_tf_fullbptt_ipc01_s0.h5 \
    checkpoints/out_tf_fullbptt_ipc05_s1.h5 \
    checkpoints/out_tf_fullbptt_ipc10_s2.h5 \
    checkpoints/out_tf_fullbptt_ipc20_s3.h5 \
    checkpoints/out_tf_fullbptt_ipc50_s4.h5 \
  --seeds 0 1 2 3 4\
  --arch text_transformer \
  --hidden_dim 256 \
  --num_classes 4 \
  --lr 1e-3 \
  --epochs 1000 \
  --batch_size 32 \
  --use_cuda

echo "Evaluation job for Transformer Full-BPTT IPC 1/5/10/20/50 submitted."
