#!/bin/bash
#SBATCH --job-name=mrpc_step1_burst
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=g2-standard-12      # 1 L4 GPU partition from the instructions
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00                 # matches the example; increase only if allowed
#SBATCH --output=mrpc_step1_burst.out
#SBATCH --error=mrpc_step1_burst.err

# Clean module environment (Burst generally has fewer modules than Greene)
module purge

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /scratch/zz3645/miniforge/etc/profile.d/conda.sh
conda activate nlp_env

echo "Hostname: $(hostname)"
echo "Python is:"
which python
python --version

echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

python main.py \
  --root ./scripts \
  --dataset mrpc_emb \
  --arch text_mlp \
  --inner_optim Adam \
  --inner_lr 0.001 \
  --lr 0.001 \
  --num_per_class 5 \
  --batch_per_class 5 \
  --task_sampler_nc 2 \
  --window 20 \
  --minwindow 0 \
  --totwindow 20 \
  --num_train_eval 4 \
  --batch_size 200 \
  --epochs 100 \
  --syn_strategy none \
  --real_strategy none \
  --fname mrpc_mlp_ipc5 \
  --name mrpc_step1_greene \
  --seed 0
