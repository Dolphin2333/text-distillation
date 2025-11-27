#!/bin/bash
#SBATCH --job-name=mrpc_step5_debug
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=debug_s1.out
#SBATCH --error=debug_s1.err

module purge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /scratch/zz3645/miniforge/etc/profile.d/conda.sh
conda activate nlp_env

echo "Hostname: $(hostname)"
which python
python --version
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

cd /scratch/zz3645/SmallD_SDD_all_progress/SmallD_SDD_InProgress

python main.py \
  --root ./scripts \
  --dataset mrpc_emb \
  --arch text_transformer \
  --inner_optim Adam \
  --inner_lr 0.001 \
  --lr 0.001 \
  --outer_optim Adam \
  --width 256 \
  --num_per_class 10 \
  --stage 1 \
  --batch_per_class 5 \
  --task_sampler_nc 2 \
  --window 20 \
  --minwindow 0 \
  --totwindow 20 \
  --num_train_eval 4 \
  --batch_size 200 \
  --epochs 0 \
  --ddtype standard \
  --syn_strategy none \
  --real_strategy none \
  --fname mrpc_ipc10_debug \
  --name mrpc_ipc10_debug \
  --seed 0 \
  --boost_dd \
  --boost_init_from out_step5_ipc5_s0_tf_adamlr.h5 \
  --boost_beta 0.0
