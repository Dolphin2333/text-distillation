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

python eval_step4_mrpc.py \
  --paths out_step3_ipc05_s0_adamlr.h5 \
          out_step3_ipc10_s1_adamlr.h5 \
          out_step3_ipc15_s2_adamlr.h5 \
          out_step3_ipc20_s3_adamlr.h5 \
          out_step3_ipc25_s4_adamlr.h5 \
          out_step3_ipc30_s5_adamlr.h5 \
  --seeds 0 1 2 3 4
