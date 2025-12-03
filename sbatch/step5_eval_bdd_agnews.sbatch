#!/bin/bash
#SBATCH --job-name=agnews_step5_eval_bdd
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/step5_eval_bdd.out
#SBATCH --error=logs/step5_eval_bdd.err

source ~/.bashrc
conda activate textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

python eval_step5_train.py \
  --data_root scripts/agnews_emb \
  --val_emb_file agnews_val_emb.pt \
  --val_label_file agnews_val_labels.pt \
  --syn_ckpt out_step5_agnews_tf_ipc20_s3.h5 \
  --ipc 20 \
  --arch text_transformer \
  --hidden_dim 256 \
  --lr 1e-3 \
  --epochs 1000 \
  --batch_size 32 \
  --seed 0 \
  --use_cuda
