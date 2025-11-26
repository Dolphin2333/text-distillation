#!/bin/bash
#SBATCH --job-name=mrpc_step3_full_boostdd
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=g2-standard-12      # 1 L4 GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=mrpc_step3_full_boostdd.out
#SBATCH --error=mrpc_step3_full_boostdd.err

module purge

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /scratch/zz3645/miniforge/etc/profile.d/conda.sh
conda activate nlp_env

echo "Hostname: $(hostname)"
echo "Python is:"
which python
python --version

echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

cd /scratch/zz3645/SmallD_SDD_all_progress/SmallD_SDD_WORKING

# Make a logs directory for per-stage outputs
mkdir -p logs

########################################
# Stage 0: IPC = 5 (no Boost-DD)
########################################
echo "===== Stage 0: IPC=5 (no Boost-DD) ====="

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
  --fname mrpc_mlp_ipc5_s0 \
  --name mrpc_step3_stage0 \
  --seed 0 \
  > logs/ipc5_s0.out 2> logs/ipc5_s0.err


########################################
# Stage 1: IPC = 10, Boost-DD from Stage 0
########################################
echo "===== Stage 1: IPC=10 (Boost-DD from Stage 0) ====="

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
  --fname mrpc_mlp_ipc10_s1 \
  --name mrpc_step3_stage1 \
  --seed 0 \
  --boost_dd \
  --boost_init_from out_step2_mrpc_emb_text_mlp_ipc5_s0.h5 \
  --boost_beta 0.0 \
  > logs/ipc10_s1.out 2> logs/ipc10_s1.err


########################################
# Stage 2: IPC = 15, Boost-DD from Stage 1
########################################
echo "===== Stage 2: IPC=15 (Boost-DD from Stage 1) ====="

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
  --fname mrpc_mlp_ipc15_s2 \
  --name mrpc_step3_stage2 \
  --seed 0 \
  --boost_dd \
  --boost_init_from out_step2_mrpc_emb_text_mlp_ipc10_s1.h5 \
  --boost_beta 0.0 \
  > logs/ipc15_s2.out 2> logs/ipc15_s2.err


########################################
# Stage 3: IPC = 20, Boost-DD from Stage 2
########################################
echo "===== Stage 3: IPC=20 (Boost-DD from Stage 2) ====="

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
  --fname mrpc_mlp_ipc20_s3 \
  --name mrpc_step3_stage3 \
  --seed 0 \
  --boost_dd \
  --boost_init_from out_step2_mrpc_emb_text_mlp_ipc15_s2.h5 \
  --boost_beta 0.0 \
  > logs/ipc20_s3.out 2> logs/ipc20_s3.err


########################################
# Stage 4: IPC = 25, Boost-DD from Stage 3
########################################
echo "===== Stage 4: IPC=25 (Boost-DD from Stage 3) ====="

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
  --fname mrpc_mlp_ipc25_s4 \
  --name mrpc_step3_stage4 \
  --seed 0 \
  --boost_dd \
  --boost_init_from out_step2_mrpc_emb_text_mlp_ipc20_s3.h5 \
  --boost_beta 0.0 \
  > logs/ipc25_s4.out 2> logs/ipc25_s4.err


########################################
# Stage 5: IPC = 30, Boost-DD from Stage 4
########################################
echo "===== Stage 5: IPC=30 (Boost-DD from Stage 4) ====="

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
  --fname mrpc_mlp_ipc30_s5 \
  --name mrpc_step3_stage5 \
  --seed 0 \
  --boost_dd \
  --boost_init_from out_step2_mrpc_emb_text_mlp_ipc25_s4.h5 \
  --boost_beta 0.0 \
  > logs/ipc30_s5.out 2> logs/ipc30_s5.err


echo "All Boost-DD stages completed."
