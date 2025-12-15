#!/bin/bash
#SBATCH --job-name=agnews_step3_mlp_fullbptt
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/step3_mlp_fullbptt.out
#SBATCH --error=logs/step3_mlp_fullbptt.err

source /scratch/hz3916/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/hz3916/miniconda3/envs/textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

mkdir -p logs/step3_mlp_fullbptt

COMMON_FLAGS="--root ./scripts --dataset agnews_emb --arch text_mlp \
  --task_sampler_nc 4 --window 40 --totwindow 40 \
  --epochs 100 --batch_size 4096 --seed 0 \
  --inner_optim Adam --inner_lr 0.001 --outer_optim Adam --lr 0.001"

run_stage() {
  local ipc=$1
  local stage=$2
  local fname=$3

  echo \"===== Full BPTT Stage ${stage}: IPC=${ipc} =====\"
  python main.py \
    $COMMON_FLAGS \
    --num_per_class ${ipc} \
    --batch_per_class 5 \
    --fname ${fname} \
    --name agnews_step3_fullbptt_s${stage} \
    --stage ${stage} \
    > logs/step3_mlp_fullbptt/${fname}.out \
    2> logs/step3_mlp_fullbptt/${fname}.err
}

run_stage 1 0 agnews_mlp_fullbptt_ipc01_s0
run_stage 10 1 agnews_mlp_fullbptt_ipc10_s1
run_stage 50 2 agnews_mlp_fullbptt_ipc50_s2

echo \"Full-BPTT MLP runs submitted (IPC 1/10/50).\"
