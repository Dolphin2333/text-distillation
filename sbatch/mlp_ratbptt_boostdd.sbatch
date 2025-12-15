#!/bin/bash
#SBATCH --job-name=agnews_mlp_ratbptt_boost
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/mlp_ratbptt_boost.out
#SBATCH --error=logs/mlp_ratbptt_boost.err

source /scratch/hz3916/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/hz3916/miniconda3/envs/textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

mkdir -p logs/mlp_ratbptt_boost

COMMON_FLAGS="--root ./scripts --dataset agnews_emb --arch text_mlp \
  --task_sampler_nc 4 --window 40 --totwindow 100 \
  --epochs 200 --batch_size 4096 --seed 0 --num_train_eval 10 \
  --inner_optim Adam --inner_lr 0.001 --outer_optim Adam --lr 0.001"

run_stage() {
  local ipc=$1
  local stage=$2
  local fname=$3
  local boost_from=$4
  local batch_pc=$5

  echo \"===== RaT-BPTT + Boost-DD Stage ${stage}: IPC=${ipc} =====\"
  python main.py \
    $COMMON_FLAGS \
    --num_per_class ${ipc} \
    --batch_per_class ${batch_pc} \
    --fname ${fname} \
    --name agnews_ratbptt_boost_s${stage} \
    --stage ${stage} \
    ${boost_from:+--boost_dd --boost_init_from ${boost_from} --boost_beta 0.3} \
    > logs/mlp_ratbptt_boost/${fname}.out \
    2> logs/mlp_ratbptt_boost/${fname}.err
}

run_stage 1 0 agnews_mlp_ratbptt_boost_ipc01_s0 '' 1
run_stage 5 1 agnews_mlp_ratbptt_boost_ipc05_s1 agnews_mlp_ratbptt_boost_ipc01_s0.h5 5
run_stage 10 2 agnews_mlp_ratbptt_boost_ipc10_s2 agnews_mlp_ratbptt_boost_ipc05_s1.h5 5
run_stage 15 3 agnews_mlp_ratbptt_boost_ipc15_s3 agnews_mlp_ratbptt_boost_ipc10_s2.h5 5
run_stage 20 4 agnews_mlp_ratbptt_boost_ipc20_s4 agnews_mlp_ratbptt_boost_ipc15_s3.h5 10
run_stage 50 5 agnews_mlp_ratbptt_boost_ipc50_s5 agnews_mlp_ratbptt_boost_ipc20_s4.h5 10

echo \"RaT-BPTT + Boost-DD MLP runs submitted (IPC 1→5→10→15→20→50).\"
