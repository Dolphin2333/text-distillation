#!/bin/bash
#SBATCH --job-name=agnews_tf_eval_base
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/step5_eval_base_tf.out
#SBATCH --error=logs/step5_eval_base_tf.err

source /scratch/hz3916/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/hz3916/miniconda3/envs/textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

python eval_step5_baseline_random_agnews.py \
  --data_root scripts/agnews_emb \
  --train_emb_file agnews_train_emb.pt \
  --train_label_file agnews_train_labels.pt \
  --val_emb_file agnews_val_emb.pt \
  --val_label_file agnews_val_labels.pt \
  --ipcs 1 5 10 15 20 25 30 50 \
  --seeds 0 1 2 3 4\
  --arch text_transformer \
  --hidden_dim 256 \
  --lr 1e-3 \
  --epochs 200 \
  --batch_size 32 \
  --use_cuda
