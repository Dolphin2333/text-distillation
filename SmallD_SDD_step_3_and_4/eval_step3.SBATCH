#!/bin/bash
#SBATCH --job-name=mrpc_eval_boostdd
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=g2-standard-12      # 1 L4 GPU partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=eval_bdd_adamlr.out
#SBATCH --error=eval_bdd_adamlr.err

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

python eval_step3.py
