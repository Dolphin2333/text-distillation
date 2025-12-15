#!/bin/bash
#SBATCH --job-name=agnews_mlp_ratbptt
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mlp_ratbptt.out
#SBATCH --error=logs/mlp_ratbptt.err

source /scratch/hz3916/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/hz3916/miniconda3/envs/textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

mkdir -p logs/mlp_ratbptt

COMMON_FLAGS="--root ./scripts --dataset agnews_emb --arch text_mlp \
  --task_sampler_nc 4 --window 40 --totwindow 100 \
  --epochs 200 --batch_size 4096 --seed 0 --num_train_eval 10 \
  --inner_optim Adam --inner_lr 0.001 --outer_optim Adam --lr 0.001"

run_stage() {
  local ipc=$1
  local stage=$2
  local fname=$3
  local batch_pc=$4

  echo \"===== RaT-BPTT Stage ${stage}: IPC=${ipc} =====\"
  python main.py \
    $COMMON_FLAGS \
    --num_per_class ${ipc} \
    --batch_per_class ${batch_pc} \
    --fname ${fname} \
    --name agnews_ratbptt_s${stage} \
    --stage ${stage} \
    > logs/mlp_ratbptt/${fname}.out \
    2> logs/mlp_ratbptt/${fname}.err
}

run_stage 1 0 agnews_mlp_ratbptt_ipc01_s0_20,40 1
run_stage 10 1 agnews_mlp_ratbptt_ipc10_s1 5
run_stage 50 2 agnews_mlp_ratbptt_ipc50_s2 10

echo \"RaT-BPTT (window 40 / total 100) MLP runs submitted (IPC 1/10/50).\"
