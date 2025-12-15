#!/bin/bash
#SBATCH --job-name=agnews_tf_ratbptt_ipc20_w20t40
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tf_ratbptt_ipc20_w20t40.out
#SBATCH --error=logs/tf_ratbptt_ipc20_w20t40.err

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
  --lr 0.0012 \
  --task_sampler_nc 5 \
  --window 20 \
  --totwindow 40 \
  --batch_size 3072 \
  --epochs 170 \
  --num_train_eval 2 \
  --ddtype standard \
  --syn_strategy none \
  --real_strategy none \
  --out_dir ./checkpoints \
  --seed 0"

echo "===== TF RaT-BPTT IPC=20 (w20/t40) ====="
python main.py \
  $COMMON_FLAGS \
  --num_per_class 20 \
  --batch_per_class 12 \
  --stage 3 \
  --fname out_tf_ratbptt_ipc20_s3_w20t40 \
  --name agnews_tf_ratbptt_s3_w20t40 \
  > logs/tf_ratbptt/out_tf_ratbptt_ipc20_s3_w20t40.out \
  2> logs/tf_ratbptt/out_tf_ratbptt_ipc20_s3_w20t40.err

echo "Submitted TF RaT-BPTT IPC=20 w20/t40 job."
