#!/bin/bash
#SBATCH --job-name=mrpc_embed
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --time=1:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=zz3645@nyu.edu
#SBATCH --output=mrpc_embed.out

module purge
module load anaconda3/2020.07


export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
conda activate /scratch/zz3645/nlp_project/nlp_env

export PATH=/scratch/zz3645/nlp_project/nlp_env/bin:$PATH

echo "Python is:"
which python
python --version

python prepare_mrpc_embeddings.py
