#!/bin/bash
#SBATCH --job-name=step5_eval_tf_all
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=step5_eval_tf_all.out
#SBATCH --error=step5_eval_tf_all.err

module purge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

source /scratch/zz3645/miniforge/etc/profile.d/conda.sh
conda activate nlp_env

cd /scratch/zz3645/SmallD_SDD_all_progress/SmallD_SDD_InProgress

mkdir -p logs_step5_eval_tf

DATA_ROOT=./scripts/mrpc_emb

########################################
# Helper: run eval for one config
# - writes full log to logs_step5_eval_tf/<tag>.out/.err
# - ALSO prints one summary line with accuracy to main stdout
########################################
run_eval () {
  local syn_ckpt=$1
  local ipc=$2
  local tag=$3  # for log filename

  echo ">>> [RUN] syn_ckpt=${syn_ckpt}, ipc=${ipc}, tag=${tag}"

  # Run eval and save detailed logs
  python eval_step5_mrpc.py \
    --data_root "${DATA_ROOT}" \
    --syn_ckpt "${syn_ckpt}" \
    --num_classes 2 \
    --ipc "${ipc}" \
    --arch text_transformer \
    --hidden_dim 256 \
    --lr 1e-3 \
    --epochs 1000 \
    --batch_size 32 \
    --seed 0 \
    --use_cuda \
    > "logs_step5_eval_tf/${tag}.out" \
    2> "logs_step5_eval_tf/${tag}.err"

  # Extract the final accuracy line from the small .out file
  local acc_line
  acc_line=$(grep "FINAL DISTILLED-SET ACCURACY" "logs_step5_eval_tf/${tag}.out" | tail -1 || echo "ACC_NOT_FOUND")

  # Print a compact summary into the MAIN sbatch .out
  echo ">>> [SUMMARY] tag=${tag}, ipc=${ipc}, ${acc_line}"
}

########################################
# β = 0 series (debug_ipc{N}_beta0)
########################################
echo "===== Evaluating β = 0 series (debug) ====="

run_eval out_step5_debug_ipc5_beta0.h5   5  "tf_beta0_ipc5"
run_eval out_step5_debug_ipc10_beta0.h5  10 "tf_beta0_ipc10"
run_eval out_step5_debug_ipc15_beta0.h5  15 "tf_beta0_ipc15"
run_eval out_step5_debug_ipc20_beta0.h5  20 "tf_beta0_ipc20"
run_eval out_step5_debug_ipc25_beta0.h5  25 "tf_beta0_ipc25"
run_eval out_step5_debug_ipc30_beta0.h5  30 "tf_beta0_ipc30"

########################################
# β ≠ 0 series (real Boost-DD)
########################################
echo "===== Evaluating β ≠ 0 Boost-DD series ====="

run_eval out_step5_ipc5_s0_tf_adamlr.h5   5  "tf_boost_ipc5"
run_eval out_step5_ipc10_s1_tf_adamlr.h5  10 "tf_boost_ipc10"
run_eval out_step5_ipc15_s2_tf_adamlr.h5  15 "tf_boost_ipc15"
run_eval out_step5_ipc20_s3_tf_adamlr.h5  20 "tf_boost_ipc20"
run_eval out_step5_ipc25_s4_tf_adamlr.h5  25 "tf_boost_ipc25"
run_eval out_step5_ipc30_s5_tf_adamlr.h5  30 "tf_boost_ipc30"

echo "All transformer evals finished."
