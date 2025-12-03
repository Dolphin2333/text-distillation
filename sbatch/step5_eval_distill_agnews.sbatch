#!/bin/bash
#SBATCH --job-name=agnews_step5_eval
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/step5_eval_distill.out
#SBATCH --error=logs/step5_eval_distill.err

source ~/.bashrc
conda activate textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

python eval_step5_distill.py \
  --data_root scripts/agnews_emb \
  --val_emb_file agnews_val_emb.pt \
  --val_label_file agnews_val_labels.pt \
  --paths "out_step5_agnews_tf_*.h5" \
  --seeds 0 1 2 \
  --arch text_transformer \
  --hidden_dim 256 \
  --lr 1e-3 \
  --epochs 1000 \
  --batch_size 32 \
  --use_cuda
