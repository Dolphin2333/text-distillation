#!/bin/bash
#SBATCH --job-name=agnews_real_mlp
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/real_mlp_baseline.out
#SBATCH --error=logs/real_mlp_baseline.err

module purge
source /scratch/hz3916/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/hz3916/miniconda3/envs/textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

python train_real_baseline.py \
  --root ./scripts/agnews_emb \
  --dataset agnews_emb \
  --arch text_mlp \
  --width 256 \
  --lr 1e-3 \
  --wd 1e-5 \
  --batch_size 2048 \
  --epochs 100 \
  --workers 4 \
  --seeds 0 1 2 3 4\
  --use_cuda

echo "Submitted real-data baseline for MLP."
