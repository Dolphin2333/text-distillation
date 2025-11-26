#!/bin/bash
#SBATCH --job-name=eval_mrpc_step1
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=eval_mrpc_step1.out
#SBATCH --error=eval_mrpc_step1.err

module purge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Miniforge + your env
source /scratch/zz3645/miniforge/etc/profile.d/conda.sh
conda activate /scratch/zz3645/conda_envs/nlp_env

echo "Hostname: $(hostname)"
echo "Python:"
which python
python --version

echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

# Assuming you submit from project root and the .pth lives there
python eval_step1_mrpc.py \
  --data_root ./scripts/mrpc_emb \
  --syn_ckpt grad_save_init_IPC_5_no_curr_unroll_20mrpc_mlp_ipc5.pth \
  --ipc 5 \
  --num_classes 2 \
  --hidden_dim 64 \
  --lr 0.001 \
  --epochs 1000 \
  --batch_size 32 \
  --seed 0 \
  --use_cuda
