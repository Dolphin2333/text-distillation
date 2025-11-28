#!/bin/bash
#SBATCH --job-name=step5_distill_eval_tf
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=step5_distill_eval_tf.out
#SBATCH --error=step5_distill_eval_tf.err

module purge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /scratch/zz3645/miniforge/etc/profile.d/conda.sh
conda activate nlp_env

cd /scratch/zz3645/SmallD_SDD_all_progress/SmallD_SDD_InProgress

# This script assumes you saved the Step-5 overview script as eval_step5_overview.py

python eval_step5_distill.py \
  --data_root ./scripts/mrpc_emb \
  --paths "out_step5_debug_ipc*_beta0.h5" "out_step5_ipc*_tf_adamlr.h5" \
  --seeds 0 1 2 3 4 \
  --arch text_transformer \
  --hidden_dim 256 \
  --lr 1e-3 \
  --epochs 1000 \
  --batch_size 32 \
  --use_cuda
