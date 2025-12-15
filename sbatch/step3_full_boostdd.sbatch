#!/bin/bash
#SBATCH --job-name=agnews_boostdd_full
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1   
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/step3_full_boostdd.out
#SBATCH --error=logs/step3_full_boostdd.err

source /scratch/hz3916/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/hz3916/miniconda3/envs/textdd


WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd $WORKDIR

mkdir -p logs/step3_full

COMMON_FLAGS="--root ./scripts --dataset agnews_emb --arch text_mlp \
  --task_sampler_nc 4 --window 40 --totwindow 120 \
  --epochs 50 --batch_size 2048 --seed 0 \
  --inner_optim Adam --inner_lr 0.001 --outer_optim Adam --lr 0.001"

run_stage() {
  local ipc=$1
  local stage=$2
  local fname=$3
  local boost_from=$4

  echo "===== Stage ${stage}: IPC=${ipc} ====="
  python main.py \
    $COMMON_FLAGS \
    --num_per_class ${ipc} \
    --batch_per_class 5 \
    --fname ${fname} \
    --name agnews_step3_s${stage} \
    --stage ${stage} \
    ${boost_from:+--boost_dd --boost_init_from ${boost_from} --boost_beta 0.3} \
    > logs/step3_full/${fname}.out \
    2> logs/step3_full/${fname}.err
}

run_stage 10 0 agnews_mlp_ipc10_s0 ''
run_stage 15 1 agnews_mlp_ipc15_s1 out_step3_agnews_mlp_ipc10_s0.h5
run_stage 20 2 agnews_mlp_ipc20_s2 out_step3_agnews_mlp_ipc15_s1.h5
run_stage 25 3 agnews_mlp_ipc25_s3 out_step3_agnews_mlp_ipc20_s2.h5
run_stage 30 4 agnews_mlp_ipc30_s4 out_step3_agnews_mlp_ipc25_s3.h5
run_stage 35 5 agnews_mlp_ipc35_s5 out_step3_agnews_mlp_ipc30_s4.h5
