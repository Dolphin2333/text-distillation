#!/bin/bash
#SBATCH --job-name=step5_boostdd_tf_adamlr
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=mrpc_step5_boostdd_tf_adamlr.out
#SBATCH --error=mrpc_step5_boostdd_tf_adamlr.err

module purge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /scratch/zz3645/miniforge/etc/profile.d/conda.sh
conda activate nlp_env   # use the env where PyTorch 2.3.1 + transformer works

echo "Hostname: $(hostname)"
echo "Python path:"
which python
python --version
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# adjust this if your working dir is different
cd /scratch/zz3645/SmallD_SDD_all_progress/SmallD_SDD_InProgress

mkdir -p logs_step5_tf_adamlr

########################################
# Common flags (for readability)
########################################
COMMON_FLAGS="\
  --root ./scripts \
  --dataset mrpc_emb \
  --arch text_transformer \
  --width 256 \
  --inner_optim Adam \
  --inner_lr 0.001 \
  --outer_optim Adam \
  --lr 0.001 \
  --task_sampler_nc 2 \
  --window 20 \
  --minwindow 0 \
  --totwindow 20 \
  --num_train_eval 4 \
  --batch_size 200 \
  --epochs 100 \
  --ddtype standard \
  --syn_strategy none \
  --real_strategy none \
  --seed 0"

########################################
# Stage 0: IPC=5 (no Boost-DD yet)
########################################
echo "===== Stage 0: IPC=5 (Transformer, baseline) ====="

python main.py \
  $COMMON_FLAGS \
  --num_per_class 5 \
  --batch_per_class 5 \
  --stage 0 \
  --fname ipc5_s0_tf_adamlr \
  --name mrpc_step5_s0_tf_adamlr \
  > logs_step5_tf_adamlr/ipc5_s0_tf_adamlr.out 2> logs_step5_tf_adamlr/ipc5_s0_tf_adamlr.err

# This should create: out_step5_ipc5_s0_tf_adamlr.h5


########################################
# Stage 1: IPC=10
########################################
echo "===== Stage 1: IPC=10 (Boost-DD from Stage 0) ====="

python main.py \
  $COMMON_FLAGS \
  --num_per_class 10 \
  --batch_per_class 5 \
  --stage 1 \
  --fname ipc10_s1_tf_adamlr \
  --name mrpc_step5_s1_tf_adamlr \
  --boost_dd \
  --boost_init_from out_step5_ipc5_s0_tf_adamlr.h5 \
  --boost_beta 0.3 \
  > logs_step5_tf_adamlr/ipc10_s1_tf_adamlr.out 2> logs_step5_tf_adamlr/ipc10_s1_tf_adamlr.err

# Produces: out_step5_ipc10_s1_tf_adamlr.h5


########################################
# Stage 2: IPC=15
########################################
echo "===== Stage 2: IPC=15 (Boost-DD from Stage 1) ====="

python main.py \
  $COMMON_FLAGS \
  --num_per_class 15 \
  --batch_per_class 5 \
  --stage 2 \
  --fname ipc15_s2_tf_adamlr \
  --name mrpc_step5_s2_tf_adamlr \
  --boost_dd \
  --boost_init_from out_step5_ipc10_s1_tf_adamlr.h5 \
  --boost_beta 0.3 \
  > logs_step5_tf_adamlr/ipc15_s2_tf_adamlr.out 2> logs_step5_tf_adamlr/ipc15_s2_tf_adamlr.err

# Produces: out_step5_ipc15_s2_tf_adamlr.h5


########################################
# Stage 3: IPC=20
########################################
echo "===== Stage 3: IPC=20 (Boost-DD from Stage 2) ====="

python main.py \
  $COMMON_FLAGS \
  --num_per_class 20 \
  --batch_per_class 5 \
  --stage 3 \
  --fname ipc20_s3_tf_adamlr \
  --name mrpc_step5_s3_tf_adamlr \
  --boost_dd \
  --boost_init_from out_step5_ipc15_s2_tf_adamlr.h5 \
  --boost_beta 0.3 \
  > logs_step5_tf_adamlr/ipc20_s3_tf_adamlr.out 2> logs_step5_tf_adamlr/ipc20_s3_tf_adamlr.err

# Produces: out_step5_ipc20_s3_tf_adamlr.h5


########################################
# Stage 4: IPC=25
########################################
echo "===== Stage 4: IPC=25 (Boost-DD from Stage 3) ====="

python main.py \
  $COMMON_FLAGS \
  --num_per_class 25 \
  --batch_per_class 5 \
  --stage 4 \
  --fname ipc25_s4_tf_adamlr \
  --name mrpc_step5_s4_tf_adamlr \
  --boost_dd \
  --boost_init_from out_step5_ipc20_s3_tf_adamlr.h5 \
  --boost_beta 0.3 \
  > logs_step5_tf_adamlr/ipc25_s4_tf_adamlr.out 2> logs_step5_tf_adamlr/ipc25_s4_tf_adamlr.err

# Produces: out_step5_ipc25_s4_tf_adamlr.h5


########################################
# Stage 5: IPC=30
########################################
echo "===== Stage 5: IPC=30 (Boost-DD from Stage 4) ====="

python main.py \
  $COMMON_FLAGS \
  --num_per_class 30 \
  --batch_per_class 5 \
  --stage 5 \
  --fname ipc30_s5_tf_adamlr \
  --name mrpc_step5_s5_tf_adamlr \
  --boost_dd \
  --boost_init_from out_step5_ipc25_s4_tf_adamlr.h5 \
  --boost_beta 0.3 \
  > logs_step5_tf_adamlr/ipc30_s5_tf_adamlr.out 2> logs_step5_tf_adamlr/ipc30_s5_tf_adamlr.err


echo "All Boost-DD Step 5 (Transformer + Adam LR) jobs completed."
