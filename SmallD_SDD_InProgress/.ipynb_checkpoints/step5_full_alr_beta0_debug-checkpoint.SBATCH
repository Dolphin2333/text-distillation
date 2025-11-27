#!/bin/bash
#SBATCH --job-name=step5_boostdd_tf_beta0
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=mrpc_step5_tf_debug_beta0.out
#SBATCH --error=mrpc_step5_tf_debug_beta0.err

module purge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /scratch/zz3645/miniforge/etc/profile.d/conda.sh
conda activate nlp_env

echo "Hostname: $(hostname)"
echo "Python path:"
which python
python --version
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd /scratch/zz3645/SmallD_SDD_all_progress/SmallD_SDD_InProgress

mkdir -p logs_step5_tf_beta0

########################################
# Common flags
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
# Stage 0: IPC=5
########################################
echo "===== Stage 0: IPC=5 ====="

python main.py \
  $COMMON_FLAGS \
  --num_per_class 5 \
  --batch_per_class 5 \
  --stage 0 \
  --fname debug_ipc5_beta0 \
  --name debug_ipc5_beta0 \
  > logs_step5_tf_beta0/debug_ipc5_beta0.out \
  2> logs_step5_tf_beta0/debug_ipc5_beta0.err

# Produces out_step5_debug_ipc5_beta0.h5


########################################
# Stage 1: IPC=10
########################################
echo "===== Stage 1: IPC=10 (Warm-start from IPC=5) ====="

python main.py \
  $COMMON_FLAGS \
  --num_per_class 10 \
  --batch_per_class 5 \
  --stage 1 \
  --fname debug_ipc10_beta0 \
  --name debug_ipc10_beta0 \
  --boost_dd \
  --boost_init_from out_step5_debug_ipc5_beta0.h5 \
  --boost_beta 0 \
  > logs_step5_tf_beta0/debug_ipc10_beta0.out \
  2> logs_step5_tf_beta0/debug_ipc10_beta0.err

# Produces out_step5_debug_ipc10_beta0.h5


########################################
# Stage 2: IPC=15
########################################
echo "===== Stage 2: IPC=15 (Warm-start from IPC=10) ====="

python main.py \
  $COMMON_FLAGS \
  --num_per_class 15 \
  --batch_per_class 5 \
  --stage 2 \
  --fname debug_ipc15_beta0 \
  --name debug_ipc15_beta0 \
  --boost_dd \
  --boost_init_from out_step5_debug_ipc10_beta0.h5 \
  --boost_beta 0 \
  > logs_step5_tf_beta0/debug_ipc15_beta0.out \
  2> logs_step5_tf_beta0/debug_ipc15_beta0.err

# Produces out_step5_debug_ipc15_beta0.h5


########################################
# Stage 3: IPC=20
########################################
echo "===== Stage 3: IPC=20 (Warm-start from IPC=15) ====="

python main.py \
  $COMMON_FLAGS \
  --num_per_class 20 \
  --batch_per_class 5 \
  --stage 3 \
  --fname debug_ipc20_beta0 \
  --name debug_ipc20_beta0 \
  --boost_dd \
  --boost_init_from out_step5_debug_ipc15_beta0.h5 \
  --boost_beta 0 \
  > logs_step5_tf_beta0/debug_ipc20_beta0.out \
  2> logs_step5_tf_beta0/debug_ipc20_beta0.err

# Produces out_step5_debug_ipc20_beta0.h5


########################################
# Stage 4: IPC=25
########################################
echo "===== Stage 4: IPC=25 (Warm-start from IPC=20) ====="

python main.py \
  $COMMON_FLAGS \
  --num_per_class 25 \
  --batch_per_class 5 \
  --stage 4 \
  --fname debug_ipc25_beta0 \
  --name debug_ipc25_beta0 \
  --boost_dd \
  --boost_init_from out_step5_debug_ipc20_beta0.h5 \
  --boost_beta 0 \
  > logs_step5_tf_beta0/debug_ipc25_beta0.out \
  2> logs_step5_tf_beta0/debug_ipc25_beta0.err

# Produces out_step5_debug_ipc25_beta0.h5


########################################
# Stage 5: IPC=30
########################################
echo "===== Stage 5: IPC=30 (Warm-start from IPC=25) ====="

python main.py \
  $COMMON_FLAGS \
  --num_per_class 30 \
  --batch_per_class 5 \
  --stage 5 \
  --fname debug_ipc30_beta0 \
  --name debug_ipc30_beta0 \
  --boost_dd \
  --boost_init_from out_step5_debug_ipc25_beta0.h5 \
  --boost_beta 0 \
  > logs_step5_tf_beta0/debug_ipc30_beta0.out \
  2> logs_step5_tf_beta0/debug_ipc30_beta0.err

# Produces out_step5_debug_ipc30_beta0.h5


echo "All Boost-DD Step 5 debug (Transformer, beta=0) jobs completed."
