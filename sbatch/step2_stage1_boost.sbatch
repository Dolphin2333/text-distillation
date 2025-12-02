#!/bin/bash
#SBATCH --job-name=agnews_stage1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/step2_stage1.out
#SBATCH --error=logs/step2_stage1.err

source ~/.bashrc
conda activate textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

python main.py \
  --root ./scripts \
  --dataset agnews_emb \
  --arch text_mlp \
  --num_per_class 15 \
  --batch_per_class 5 \
  --task_sampler_nc 4 \
  --window 40 \
  --totwindow 120 \
  --epochs 2000 \
  --batch_size 200 \
  --fname agnews_mlp_ipc15_s1 \
  --name agnews_step3_s1 \
  --boost_dd \
  --boost_init_from out_step3_agnews_mlp_ipc10_s0.h5 \
  --boost_beta 0.3 \
  --stage 1 \
  --seed 0
