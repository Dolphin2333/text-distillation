#!/bin/bash
#SBATCH --job-name=mrpc_step5_eval_tf
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=mrpc_step5_eval_tf.out
#SBATCH --error=mrpc_step5_eval_tf.err

module purge

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /scratch/zz3645/miniforge/etc/profile.d/conda.sh
conda activate nlp_env   # same env as training

echo "Hostname: $(hostname)"
which python
python --version
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

python eval_step5_train.py \
  --data_root ./scripts/mrpc_emb \
  --syn_ckpt out_step5_mrpc_tf_ipc5_try2.pth \
  --num_classes 2 \
  --ipc 5 \
  --arch text_transformer \
  --hidden_dim 256 \
  --lr 1e-3 \
  --epochs 1000 \
  --batch_size 32 \
  --seed 0 \
  --use_cuda
