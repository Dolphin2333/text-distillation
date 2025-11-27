#!/bin/bash
#SBATCH --job-name=mrpc_step3_stage3
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=mrpc_step3_stage3.out
#SBATCH --error=mrpc_step3_stage3.err

module purge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /scratch/zz3645/miniforge/etc/profile.d/conda.sh
conda activate nlp_env

echo "Hostname: $(hostname)"
which python
python --version
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

cd /scratch/zz3645/SmallD_SDD_all_progress/SmallD_SDD_WORKING

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
  --boost_beta 0.0
