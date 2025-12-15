#!/bin/bash
#SBATCH --job-name=agnews_tf_fullbptt_ipc50_boost
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/tf_fullbptt_ipc50_boost.out
#SBATCH --error=logs/tf_fullbptt_ipc50_boost.err

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
  --epochs 200 \
  --num_train_eval 4 \
  --ddtype standard \
  --syn_strategy none \
  --real_strategy none \
  --out_dir ./checkpoints \
  --seed 0"

BOOST_FLAGS="\
  --boost_dd \
  --boost_init_from checkpoints/out_tf_fullbptt_ipc20_s3.h5"

echo "===== TF Full BPTT IPC=50 (boosted stage 4) ====="
python main.py \
  $COMMON_FLAGS \
  $BOOST_FLAGS \
  --num_per_class 50 \
  --batch_per_class 20 \
  --stage 4 \
  --fname out_tf_fullbptt_ipc50_s4_boost \
  --name agnews_tf_fullbptt_s4_boost \
  > logs/tf_fullbptt/out_tf_fullbptt_ipc50_s4_boost.out \
  2> logs/tf_fullbptt/out_tf_fullbptt_ipc50_s4_boost.err

echo "Submitted TF Full-BPTT IPC=50 Boost-DD job."
