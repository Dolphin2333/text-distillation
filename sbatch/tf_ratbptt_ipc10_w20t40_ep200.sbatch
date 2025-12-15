#!/bin/bash
#SBATCH --job-name=agnews_tf_ratbptt_ipc10_w20t40_ep200
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tf_ratbptt_ipc10_w20t40_ep200.out
#SBATCH --error=logs/tf_ratbptt_ipc10_w20t40_ep200.err

module purge
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

source /scratch/hz3916/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/hz3916/miniconda3/envs/textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

mkdir -p logs/tf_ratbptt

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
  --window 20 \
  --totwindow 40 \
  --batch_size 2048 \
  --epochs 200 \
  --num_train_eval 2 \
  --ddtype standard \
  --syn_strategy none \
  --real_strategy none \
  --out_dir ./checkpoints \
  --seed 0"

echo "===== TF RaT-BPTT IPC=10 (w20/t40, 200 epochs) ====="
python main.py \
  $COMMON_FLAGS \
  --num_per_class 10 \
  --batch_per_class 8 \
  --stage 2 \
  --fname out_tf_ratbptt_ipc10_s2_w20t40_ep200 \
  --name agnews_tf_ratbptt_s2_w20t40_ep200 \
  > logs/tf_ratbptt/out_tf_ratbptt_ipc10_s2_w20t40_ep200.out \
  2> logs/tf_ratbptt/out_tf_ratbptt_ipc10_s2_w20t40_ep200.err

echo "Submitted TF RaT-BPTT IPC=10 w20/t40 200-epoch job."
