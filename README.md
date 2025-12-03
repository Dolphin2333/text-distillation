# AG News Distillation Pipeline (aligned with Tim's code)

Use these sbatch wrappers to replicate Tim's MRPC experiments on a new embedding dataset (AG News). The `tim-code/` folder is a worktree checkout of the original `origin/Tim` branch; only minimal additions were made to register `agnews_emb` and provide an embedding script.

## Step-by-step workflow (HPC)

1. **Prepare embeddings** – generates CLS tensors for both Step3 (MLP) and Step5 (Transformer).
   ```bash
   sbatch sbatch/step0_prepare_embeddings.sbatch
   ```
   This runs the embedding scripts inside `tim-code/SmallD_SDD_step_3_and_4/scripts/` and `tim-code/SmallD_SDD_InProgress/scripts/`, writing `.pt` files to each directory's `scripts/agnews_emb/`.

2. **Stage 0 (RaT-BPTT + TextMLP)**
   ```bash
   sbatch sbatch/step1_stage0_mlp.sbatch
   ```
   Executes `tim-code/SmallD_SDD_step_3_and_4/main.py` with `--dataset agnews_emb`, IPC=10. Output checkpoint: `tim-code/SmallD_SDD_step_3_and_4/out_step3_agnews_mlp_ipc10_s0.h5`.

3. **Boost-DD stages (TextMLP)**
   - Run Stage 1 only: `sbatch sbatch/step2_stage1_boost.sbatch` (IPC=15, warm-start from Stage0).
   - Or run the entire schedule (IPC 10→35) via `sbatch sbatch/step3_full_boostdd.sbatch`. All `out_step3_agnews_*.h5` files stay under `tim-code/SmallD_SDD_step_3_and_4/`.

4. **Evaluate MLP distilled sets / random baseline**
   - Distilled-set eval: `sbatch sbatch/step4_eval_distilled.sbatch` → trains `TextMLP` students on each `out_step3_agnews_*.h5`, reports mean±std accuracy (`logs/step4_eval_syn.out`).
   - Random baseline: `sbatch sbatch/step4_eval_baseline.sbatch` → draws real subsets with same IPC.

5. **Transformer + Boost-DD (Step5)**
   ```bash
   sbatch sbatch/step5_full_transformer.sbatch
   ```
   Runs `tim-code/SmallD_SDD_InProgress/main.py` with `--arch text_transformer` across stages IPC=5→30. Outputs stored in `tim-code/SmallD_SDD_InProgress/out_step5_agnews_tf_*.h5`.

6. **Evaluate Transformer distilled sets**
   - Distilled: `sbatch sbatch/step5_eval_distill.sbatch` (calls `eval_step5_distill.py`, prints tables under `logs/step5_eval_distill.out`).
   - Random baseline: `sbatch sbatch/step5_eval_baseline.sbatch`.

## Notes
- All sbatch scripts assume the repo lives at `/scratch/hz3916/Data_Distillation/text-distillation`; adjust `WORKDIR` if different.
- Tim's MLP/Transformer code remains unchanged except for dataset registration. You can still run MRPC by pointing `--dataset mrpc_emb` and using the original `.pt` files.
- Evaluation scripts now accept `--data_root/--train_emb/...` so you can switch between MRPC and AG News via CLI.
- Distilled checkpoints and logs are inside `tim-code/SmallD_SDD_*` directories (not the repo root). Keep this in mind when collecting results.
