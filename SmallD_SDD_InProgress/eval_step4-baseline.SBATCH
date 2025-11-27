#!/bin/bash
#SBATCH --job-name=mrpc_step4_eval
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=mrpc_step4_eval.out
#SBATCH --error=mrpc_step4_eval.err

module purge
source /scratch/zz3645/miniforge/etc/profile.d/conda.sh
conda activate nlp_env

cd /scratch/zz3645/SmallD_SDD_all_progress/SmallD_SDD_WORKING

python eval_baseline_random_mrpc.py --ipcs 5 10 15 20 25 30 --seeds 0 1 2 3 4

