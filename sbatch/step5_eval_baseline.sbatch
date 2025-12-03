#!/bin/bash
#SBATCH --job-name=agnews_step5_baseline
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/step5_eval_baseline.out
#SBATCH --error=logs/step5_eval_baseline.err

source ~/.bashrc
conda activate textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation/tim-code/SmallD_SDD_InProgress
cd "$WORKDIR"

python eval_step5_baseline_random_mrpc.py \
  --data_root scripts/agnews_emb \
  --train_emb_file agnews_train_emb.pt \
  --train_label_file agnews_train_labels.pt \
  --val_emb_file agnews_val_emb.pt \
  --val_label_file agnews_val_labels.pt \
  --ipcs 5 10 15 20 25 30 \
  --seeds 0 1 2 \
  --arch text_transformer \
  --hidden_dim 256 \
  --lr 1e-3 \
  --epochs 200 \
  --batch_size 32 \
  --use_cuda
