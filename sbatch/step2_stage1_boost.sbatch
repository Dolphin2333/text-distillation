#!/bin/bash
#SBATCH --job-name=step2_stage1_boost_agnews
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1   
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/step2_stage1.out
#SBATCH --error=logs/step2_stage1.err

source /scratch/hz3916/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/hz3916/miniconda3/envs/textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd $WORKDIR

python main.py \
  --root ./scripts \
  --dataset agnews_emb \
  --arch text_mlp \
  --num_per_class 15 \
  --batch_per_class 5 \
  --task_sampler_nc 4 \
  --window 40 \
  --totwindow 120 \
  --epochs 30 \
  --batch_size 2048 \
  --fname agnews_mlp_ipc15_s1 \
  --name agnews_step3_s1 \
  --boost_dd \
  --boost_init_from out_step3_agnews_mlp_ipc10_s0.h5 \
  --boost_beta 0.3 \
  --stage 1 \
  --seed 0
