#!/bin/bash
#SBATCH --job-name=agnews_tf_ratbptt
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/tf_ratbptt.out
#SBATCH --error=logs/tf_ratbptt.err

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
  --window 40 \
  --totwindow 100 \
  --batch_size 4096 \
  --epochs 200 \
  --num_train_eval 4 \
  --ddtype standard \
  --syn_strategy none \
  --real_strategy none \
  --out_dir ./checkpoints \
  --seed 0"

run_stage() {
  local ipc=$1
  local stage=$2
  local fname=$3
  local batch_pc=$4

  echo "===== TF RaT-BPTT Stage ${stage}: IPC=${ipc} ====="
  python main.py \
    $COMMON_FLAGS \
    --num_per_class "${ipc}" \
    --batch_per_class "${batch_pc}" \
    --stage "${stage}" \
    --fname "${fname}" \
    --name agnews_tf_ratbptt_s${stage} \
    > "logs/tf_ratbptt/${fname}.out" \
    2> "logs/tf_ratbptt/${fname}.err"
}

run_stage 1 0 out_tf_ratbptt_ipc01_s0 1
run_stage 5 1 out_tf_ratbptt_ipc05_s1 5
run_stage 10 2 out_tf_ratbptt_ipc10_s2 5
run_stage 20 3 out_tf_ratbptt_ipc20_s3 10
run_stage 50 4 out_tf_ratbptt_ipc50_s4 10

echo "Transformer RaT-BPTT IPC 1,5,10,20,50 runs submitted."
