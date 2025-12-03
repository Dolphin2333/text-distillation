#!/bin/bash
#SBATCH --job-name=agnews_step5_tf
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/step5_full_tf.out
#SBATCH --error=logs/step5_full_tf.err

source ~/.bashrc
conda activate textdd

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

mkdir -p logs/step5_full_tf

COMMON_FLAGS="--root ./scripts --dataset agnews_emb --arch text_transformer \
  --width 256 --inner_optim Adam --inner_lr 0.001 --outer_optim Adam --lr 0.001 \
  --task_sampler_nc 4 --window 20 --totwindow 20 --batch_size 200 --epochs 100 --seed 0"

run_stage() {
  local ipc=$1
  local stage=$2
  local fname=$3
  local boost_from=$4

  echo "===== Transformer Stage ${stage}: IPC=${ipc} ====="
  python main.py \
    $COMMON_FLAGS \
    --num_per_class ${ipc} \
    --batch_per_class 5 \
    --fname ${fname} \
    --name agnews_step5_s${stage}_tf \
    --stage ${stage} \
    ${boost_from:+--boost_dd --boost_init_from ${boost_from} --boost_beta 0.3} \
    > logs/step5_full_tf/${fname}.out \
    2> logs/step5_full_tf/${fname}.err
}

run_stage 5 0 out_step5_agnews_tf_ipc5_s0 ''
run_stage 10 1 out_step5_agnews_tf_ipc10_s1 out_step5_agnews_tf_ipc5_s0.h5
run_stage 15 2 out_step5_agnews_tf_ipc15_s2 out_step5_agnews_tf_ipc10_s1.h5
run_stage 20 3 out_step5_agnews_tf_ipc20_s3 out_step5_agnews_tf_ipc15_s2.h5
run_stage 25 4 out_step5_agnews_tf_ipc25_s4 out_step5_agnews_tf_ipc20_s3.h5
run_stage 30 5 out_step5_agnews_tf_ipc30_s5 out_step5_agnews_tf_ipc25_s4.h5
