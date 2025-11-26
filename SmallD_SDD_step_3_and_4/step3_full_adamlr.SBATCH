#!/bin/bash
#SBATCH --job-name=mrpc_step3_boostdd_adamlr
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=mrpc_step3_boostdd_adamlr.out
#SBATCH --error=mrpc_step3_boostdd_adamlr.err

module purge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /scratch/zz3645/miniforge/etc/profile.d/conda.sh
conda activate nlp_env

echo "Hostname: $(hostname)"
echo "Python path:"
which python
python --version
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

cd /scratch/zz3645/SmallD_SDD_all_progress/SmallD_SDD_WORKING

mkdir -p logs_step3_adamlr

########################################
# Stage 0: IPC=5
########################################
echo "===== Stage 0: IPC=5 ====="

python main.py \
  --root ./scripts \
  --dataset mrpc_emb \
  --arch text_mlp \
  --inner_optim Adam \
  --inner_lr 0.001 \
  --lr 0.001 \
  --num_per_class 5 \
  --stage 0 \
  --batch_per_class 5 \
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
  --fname ipc05_s0_adamlr \
  --name mrpc_step3_s0_adamlr \
  --seed 0 \
  > logs_step3_adamlr/ipc05_s0_adamlr.out 2> logs_step3_adamlr/ipc05_s0_adamlr.err


########################################
# Stage 1: IPC=10
########################################
echo "===== Stage 1: IPC=10 ====="

python main.py \
  --root ./scripts \
  --dataset mrpc_emb \
  --arch text_mlp \
  --inner_optim Adam \
  --inner_lr 0.001 \
  --lr 0.001 \
  --num_per_class 10 \
  --stage 1 \
  --batch_per_class 5 \
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
  --fname ipc10_s1_adamlr \
  --name mrpc_step3_s1_adamlr \
  --seed 0 \
  --boost_dd \
  --boost_init_from out_step3_ipc05_s0_adamlr.h5 \
  --boost_beta 0.3 \
  > logs_step3_adamlr/ipc10_s1_adamlr.out 2> logs_step3_adamlr/ipc10_s1_adamlr.err


########################################
# Stage 2: IPC=15
########################################
echo "===== Stage 2: IPC=15 ====="

python main.py \
  --root ./scripts \
  --dataset mrpc_emb \
  --arch text_mlp \
  --inner_optim Adam \
  --inner_lr 0.001 \
  --lr 0.001 \
  --num_per_class 15 \
  --stage 2 \
  --batch_per_class 5 \
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
  --fname ipc15_s2_adamlr \
  --name mrpc_step3_s2_adamlr \
  --seed 0 \
  --boost_dd \
  --boost_init_from out_step3_ipc10_s1_adamlr.h5 \
  --boost_beta 0.3 \
  > logs_step3_adamlr/ipc15_s2_adamlr.out 2> logs_step3_adamlr/ipc15_s2_adamlr.err


########################################
# Stage 3: IPC=20
########################################
echo "===== Stage 3: IPC=20 ====="

python main.py \
  --root ./scripts \
  --dataset mrpc_emb \
  --arch text_mlp \
  --inner_optim Adam \
  --inner_lr 0.001 \
  --lr 0.001 \
  --num_per_class 20 \
  --stage 3 \
  --batch_per_class 5 \
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
  --fname ipc20_s3_adamlr \
  --name mrpc_step3_s3_adamlr \
  --seed 0 \
  --boost_dd \
  --boost_init_from out_step3_ipc15_s2_adamlr.h5 \
  --boost_beta 0.3 \
  > logs_step3_adamlr/ipc20_s3_adamlr.out 2> logs_step3_adamlr/ipc20_s3_adamlr.err


########################################
# Stage 4: IPC=25
########################################
echo "===== Stage 4: IPC=25 ====="

python main.py \
  --root ./scripts \
  --dataset mrpc_emb \
  --arch text_mlp \
  --inner_optim Adam \
  --inner_lr 0.001 \
  --lr 0.001 \
  --num_per_class 25 \
  --stage 4 \
  --batch_per_class 5 \
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
  --fname ipc25_s4_adamlr \
  --name mrpc_step3_s4_adamlr \
  --seed 0 \
  --boost_dd \
  --boost_init_from out_step3_ipc20_s3_adamlr.h5 \
  --boost_beta 0.3 \
  > logs_step3_adamlr/ipc25_s4_adamlr.out 2> logs_step3_adamlr/ipc25_s4_adamlr.err


########################################
# Stage 5: IPC=30
########################################
echo "===== Stage 5: IPC=30 ====="

python main.py \
  --root ./scripts \
  --dataset mrpc_emb \
  --arch text_mlp \
  --inner_optim Adam \
  --inner_lr 0.001 \
  --lr 0.001 \
  --num_per_class 30 \
  --stage 5 \
  --batch_per_class 5 \
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
  --fname ipc30_s5_adamlr \
  --name mrpc_step3_s5_adamlr \
  --seed 0 \
  --boost_dd \
  --boost_init_from out_step3_ipc25_s4_adamlr.h5 \
  --boost_beta 0.3 \
  > logs_step3_adamlr/ipc30_s5_adamlr.out 2> logs_step3_adamlr/ipc30_s5_adamlr.err


echo "All Boost-DD Stage 3 (Adam LR-mix) jobs completed."
