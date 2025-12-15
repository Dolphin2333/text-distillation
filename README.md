# Embarrassingly Simple Dataset Distillation For Natural Language Processing 📊💡

## 🔍 Project Overview
This repository accompanies the NYU CDS project **“Embarrassingly Simple Dataset Distillation for Natural Language Processing”**. Building on the RaT‑BPTT and Boost‑DD strategies from the CV literature, we distill small sets of *continuous CLS embeddings* that allow lightweight student models (Text MLP or Transformer) to match or outperform randomly sampled real subsets on MRPC and AG News.

Key ideas:
- Distillation is formulated as a bilevel problem. The inner loop trains a student model on synthetic embeddings, while the outer loop updates the embeddings via (RaT-)BPTT.
- RaT‑BPTT stabilizes gradients by randomly truncating long unrolls, keeping runtime manageable even for Transformer students.
- Boost-DD warm-starts higher IPC stages with previously distilled checkpoints, yielding more stable solutions for large synthetic budgets.
- All experiments run entirely in embedding space using frozen sentence encoders, which makes the approach “embarrassingly simple” to deploy for new NLP datasets.

## 📁 Tutorial & Repository Structure

```
text-distillation/
├── main.py                     # Unified training entry point (MLP or Transformer students)
├── framework/                  # Core distillation components (datasets, optimizers, utilities)
├── scripts/agnews_emb/         # Frozen CLS embeddings for AG News (generated via step0)
├── sbatch/                     # SLURM batch scripts for every experiment stage
├── eval_*.py                   # Evaluation utilities for distilled vs. random baselines
├── train_real_baseline.py      # Optional real-data baseline training script
└── logs/                       # Saved stdout/stderr for every sbatch job
```

### 🧩 Core Code Components
- **`main.py`** – Parses CLI flags and launches dataset distillation (supports RaT-BPTT, Full-BPTT, Boost-DD warm starts).
- **`framework/`**
  - `base.py` – Bilevel training loop, evaluation routines, checkpointing.
  - `config.py` – Model/dataset configs and scheduling utilities.
  - `util.py` – Augmentation helpers, projection utilities, Boost-DD helpers.
- **Evaluation scripts**
  - `eval_step5_distill.py` – Re-trains student models on `.h5` distilled sets and reports mean ± std accuracy across seeds.
  - `eval_step5_train.py` / `eval_step5_baseline_random_agnews.py` – Support scripts for random baseline comparisons.
- **Embedding scripts** (`sbatch/step0_prepare_embeddings.sbatch`) – Generates CLS tensors for new datasets via frozen encoders.

### Running the Pipeline (HPC Workflow)
1. **Prepare embeddings**  
   `sbatch sbatch/step0_prepare_embeddings.sbatch`
2. **Text MLP RaT-BPTT & Boost-DD stages**  
   `sbatch sbatch/mlp_ratbptt_ipc1_10_50.sbatch`, `sbatch sbatch/mlp_ratbptt_ipc5_20.sbatch`, etc.
3. **Text MLP Full-BPTT stages**  
   `sbatch sbatch/mlp_fullbptt_ipc1_10_50.sbatch`, `sbatch sbatch/mlp_fullbptt_ipc5_20.sbatch`.
4. **Transformer Full-BPTT or RaT-BPTT**  
   `sbatch sbatch/tf_fullbptt_ipc1_5_10_20_50.sbatch`, `sbatch/tf_ratbptt_ipc1_5_10_20_50.sbatch`, plus extended-epoch variants.
5. **Evaluation**  
   `sbatch sbatch/eval_tf_fullbptt_ipc1_5_10_20_50.sbatch`, `sbatch/sbatch/eval_mlp_*`, or `python eval_step5_distill.py ...`.

Each SLURM script writes `.out/.err` logs under `logs/`, and distilled checkpoints (`*.h5`, `*.pth`) reside in `./checkpoints/`.

## References
The project draws on the literature cited in the accompanying report “Small_data_Project (6).pdf”:

1. Brown, T. B., Mann, B., Ryder, N., *et al.* (2020). **Language Models are Few-Shot Learners.** *NeurIPS.*
2. Carlini, N., Tramer, F., Wallace, E., *et al.* (2021). **Extracting Training Data from Large Language Models.** *USENIX Security Symposium.*
3. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.** *NAACL.*
4. Feng, W., Huang, Y., Liu, Y., Zhang, Z., Li, Y., & Wang, L. (2023). **Embarrassingly Simple Dataset Distillation.** *ICLR.*
5. Maekawa, T., Tanaka, S., & Okazaki, N. (2023). **Dataset Distillation with Attention Labels for Fine-Tuning BERT.** *Findings of ACL.*
6. Schick, T., & Schütze, H. (2020). **It’s Not Just Size That Matters: Small Language Models are also Few-Shot Learners.** *NAACL.*
7. Wang, T., Zhu, J.-Y., Torralba, A., & Efros, A. A. (2018). **Dataset Distillation.** *CVPR.*
8. Williams, R. J., & Peng, J. (1990). **An Efficient Gradient-Based Algorithm for On-line Training of Recurrent Network Trajectories.** *Neural Computation.*

---
For full experimental details, plots, and scripted commands, refer to the `analysis/` artifacts and the PDF report included in this repository.
