#!/bin/bash
#SBATCH --job-name=mrpc_step2_grid
#SBATCH --account=ds_ga_3001_003-2025fa
#SBATCH --partition=g2-standard-12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-5
#SBATCH --output=logs/step2_grid_%A_%a.out
#SBATCH --error=logs/step2_grid_%A_%a.err

module purge
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate your env
source /scratch/zz3645/miniforge/etc/profile.d/conda.sh
conda activate /scratch/zz3645/conda_envs/nlp_env

echo "Hostname: $(hostname)"
echo "Python:"
which python
python --version
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"

# ---- Hyperparam grid definition ----

IPCS=(1 5 10)
WINDOWS=(10 20)

TASK_ID=${SLURM_ARRAY_TASK_ID}

# Derive which ipc / window this task uses
NUM_IPC=${#IPCS[@]}          # 3
NUM_WIN=${#WINDOWS[@]}       # 2

IPC_INDEX=$(( TASK_ID % NUM_IPC ))
WIN_INDEX=$(( TASK_ID / NUM_IPC ))

IPC=${IPCS[$IPC_INDEX]}
WIN=${WINDOWS[$WIN_INDEX]}

SEED=0
INNER_LR=0.001
OUTER_LR=0.001
TASK_SAMPLER_NC=2            # for MRPC’s 2 classes

echo "Task ID: $TASK_ID"
echo "Config: IPC=${IPC}, window=${WIN}, seed=${SEED}"

# ---- Unique experiment name & fname ----
EXP_TAG="ipc${IPC}_w${WIN}_seed${SEED}"
FNAME="mrpc_mlp_${EXP_TAG}"

# Where scripts live relative to project root
DATA_ROOT="./scripts"

# 1) Run training (RaT-BPTT distillation)
python main.py \
  --root "${DATA_ROOT}" \
  --dataset mrpc_emb \
  --arch text_mlp \
  --inner_optim Adam \
  --inner_lr ${INNER_LR} \
  --lr ${OUTER_LR} \
  --num_per_class ${IPC} \
  --batch_per_class ${IPC} \
  --task_sampler_nc ${TASK_SAMPLER_NC} \
  --window ${WIN} \
  --minwindow 0 \
  --totwindow ${WIN} \
  --num_train_eval 4 \
  --batch_size 200 \
  --epochs 100 \
  --syn_strategy none \
  --real_strategy none \
  --fname ${FNAME} \
  --name "mrpc_step2_${EXP_TAG}" \
  --seed ${SEED}

TRAIN_EXIT=$?
echo "Training exit code: ${TRAIN_EXIT}"

# Path to the synthetic checkpoint that main.py should have saved
CKPT="grad_save_init_IPC_${IPC}_no_curr_unroll_${WIN}${FNAME}.pth"

if [ ${TRAIN_EXIT} -ne 0 ]; then
  echo "Training failed, not running eval. Expected ckpt: ${CKPT}"
  exit ${TRAIN_EXIT}
fi

if [ ! -f "${CKPT}" ]; then
  echo "ERROR: Expected checkpoint not found: ${CKPT}"
  ls -lh
  exit 1
fi

echo "Found checkpoint: ${CKPT}"

# 2) Evaluate distilled set using the eval script you already have
python eval_step1_mrpc.py \
  --data_root ./scripts/mrpc_emb \
  --syn_ckpt "${CKPT}" \
  --ipc ${IPC} \
  --num_classes 2 \
  --hidden_dim 64 \
  --lr 0.001 \
  --epochs 1000 \
  --batch_size 32 \
  --seed 0 \
  --use_cuda

EVAL_EXIT=$?
echo "Eval exit code: ${EVAL_EXIT}"

exit ${EVAL_EXIT}
