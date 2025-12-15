#!/bin/bash
#SBATCH --job-name=agnews_tf_fullbptt_ipc50_noboost
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --output=logs/tf_fullbptt_ipc50_noboost.out
#SBATCH --error=logs/tf_fullbptt_ipc50_noboost.err

module purge
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

source /scratch/hz3916/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/hz3916/miniconda3/envs/textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

mkdir -p logs/tf_fullbptt

COMMON_FLAGS="\
  --root ./scripts \
  --dataset agnews_emb \
  --arch text_transformer \
  --width 256 \
  --inner_optim Adam \
  --inner_lr 0.0015 \
  --outer_optim Adam \
  --lr 0.002 \
  --task_sampler_nc 6 \
  --window 40 \
  --totwindow 40 \
  --batch_size 4096 \
  --epochs 600 \
  --num_train_eval 2 \
  --ddtype standard \
  --syn_strategy none \
  --real_strategy none \
  --out_dir ./checkpoints \
  --seed 0"

echo "===== TF Full BPTT IPC=50 (no Boost-DD) ====="
python main.py \
  $COMMON_FLAGS \
  --num_per_class 50 \
  --batch_per_class 20 \
  --stage 4 \
  --fname out_tf_fullbptt_ipc50_s4_noboost \
  --name agnews_tf_fullbptt_s4_noboost \
  > logs/tf_fullbptt/out_tf_fullbptt_ipc50_s4_noboost.out \
  2> logs/tf_fullbptt/out_tf_fullbptt_ipc50_s4_noboost.err

echo "Submitted TF Full-BPTT IPC=50 no-boost job."
