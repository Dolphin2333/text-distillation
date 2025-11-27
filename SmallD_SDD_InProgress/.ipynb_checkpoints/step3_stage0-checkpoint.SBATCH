#!/bin/bash
#SBATCH --job-name=mrpc_step5_try1
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=g2-standard-12      # 1 L4 GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=out_step5_try1.out
#SBATCH --error=out_step5_try1.err

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
  --fname mrpc_mlp_ipc5_boost0 \
  --name mrpc_step3_stage0 \
  --seed 0 \
  --width 256