#!/bin/bash
#SBATCH --job-name=agnews_tf_fullbptt_ipc50_boost_ep600
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=logs/tf_fullbptt_ipc50_boost_ep600.out
#SBATCH --error=logs/tf_fullbptt_ipc50_boost_ep600.err

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
  --num_train_eval 4 \
  --ddtype standard \
  --syn_strategy none \
  --real_strategy none \
  --out_dir ./checkpoints \
  --seed 0"

BOOST_FLAGS="\
  --boost_dd \
  --boost_init_from checkpoints/out_tf_fullbptt_ipc20_s3.h5"

echo "===== TF Full BPTT IPC=50 Boost-DD (600 epochs) ====="
python main.py \
  $COMMON_FLAGS \
  $BOOST_FLAGS \
  --num_per_class 50 \
  --batch_per_class 20 \
  --stage 4 \
  --fname out_tf_fullbptt_ipc50_s4_boost_ep600 \
  --name agnews_tf_fullbptt_s4_boost_ep600 \
  > logs/tf_fullbptt/out_tf_fullbptt_ipc50_s4_boost_ep600.out \
  2> logs/tf_fullbptt/out_tf_fullbptt_ipc50_s4_boost_ep600.err

echo "Submitted TF Full-BPTT IPC=50 Boost-DD (600 epoch) job."
