#!/bin/bash
#SBATCH --job-name=agnews_stage0
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/step1_stage0.out
#SBATCH --error=logs/step1_stage0.err

source ~/.bashrc
conda activate textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation/tim-code/SmallD_SDD_step_3_and_4
cd "$WORKDIR"

python main.py \
  --root ./scripts \
  --dataset agnews_emb \
  --arch text_mlp \
  --num_per_class 10 \
  --batch_per_class 5 \
  --task_sampler_nc 4 \
  --window 40 \
  --totwindow 120 \
  --epochs 2000 \
  --batch_size 200 \
  --fname agnews_mlp_ipc10_s0 \
  --name agnews_step3_s0 \
  --seed 0
