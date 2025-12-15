#!/bin/bash
#SBATCH --job-name=agnews_tf_fullbptt_ipc50
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/tf_fullbptt_ipc50.out
#SBATCH --error=logs/tf_fullbptt_ipc50.err

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
  --inner_lr 0.001 \
  --outer_optim Adam \
  --lr 0.001 \
  --task_sampler_nc 4 \
  --window 40 \
  --totwindow 40 \
  --batch_size 2048 \
  --epochs 100 \
  --num_train_eval 4 \
  --ddtype standard \
  --syn_strategy none \
  --real_strategy none \
  --out_dir ./checkpoints \
  --seed 0"

echo "===== TF Full BPTT IPC=50 (stage 4) ====="
python main.py \
  $COMMON_FLAGS \
  --num_per_class 50 \
  --batch_per_class 10 \
  --stage 4 \
  --fname out_tf_fullbptt_ipc50_s4 \
  --name agnews_tf_fullbptt_s4 \
  > logs/tf_fullbptt/out_tf_fullbptt_ipc50_s4.out \
  2> logs/tf_fullbptt/out_tf_fullbptt_ipc50_s4.err

echo "Submitted TF Full-BPTT IPC=50 job."
