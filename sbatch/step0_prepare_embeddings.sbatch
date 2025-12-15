#!/bin/bash
#SBATCH --job-name=agnews_embed
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/step0_embed.out
#SBATCH --error=logs/step0_embed.err

source /scratch/hz3916/.bashrc
conda activate textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

python scripts/prepare_agnews_embeddings.py --batch_size 64 --device cuda
