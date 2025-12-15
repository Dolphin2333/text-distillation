#!/bin/bash
#SBATCH --job-name=agnews_step5_tf_resume
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=logs/step5_full_tf_resume.out
#SBATCH --error=logs/step5_full_tf_resume.err

module purge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /scratch/hz3916/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/hz3916/miniconda3/envs/textdd

echo "Hostname: $(hostname)"
echo "Python path:"
which python
python --version
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

WORKDIR=/scratch/hz3916/Data_Distillation/text-distillation
cd "$WORKDIR"

mkdir -p logs/step5_full_tf_resume

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
  --totwindow 20 \
  --batch_size 200 \
  --epochs 100 \
  --seed 0"

run_stage() {
  local ipc=$1
  local stage=$2
  local fname=$3
  local boost_from=$4

  echo "===== Resume Stage ${stage}: IPC=${ipc} ====="
  python main.py \
    $COMMON_FLAGS \
    --num_per_class ${ipc} \
    --batch_per_class 5 \
    --fname ${fname} \
    --name agnews_step5_s${stage}_tf \
    --stage ${stage} \
    --boost_dd \
    --boost_init_from ${boost_from} \
    --boost_beta 0.3 \
    > logs/step5_full_tf_resume/${fname}.out \
    2> logs/step5_full_tf_resume/${fname}.err
}

# Stage 1: resume from Stage 0 output
run_stage 10 1 out_step5_agnews_tf_ipc10_s1 out_step3_out_step5_agnews_tf_ipc5_s0.h5
# Stage 2: resume from stage 1 output, which will also be added a out_step3_ prefix
run_stage 15 2 out_step5_agnews_tf_ipc15_s2 out_step3_out_step5_agnews_tf_ipc10_s1.h5
# Stage 3
run_stage 20 3 out_step5_agnews_tf_ipc20_s3 out_step3_out_step5_agnews_tf_ipc15_s2.h5
# Stage 4
run_stage 25 4 out_step5_agnews_tf_ipc25_s4 out_step3_out_step5_agnews_tf_ipc20_s3.h5
# Stage 5
run_stage 30 5 out_step5_agnews_tf_ipc30_s5 out_step3_out_step5_agnews_tf_ipc25_s4.h5

echo "Resumed stages 1-5 submitted."