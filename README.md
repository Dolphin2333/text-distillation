# AG News Text Distillation 

This directory packages the most stable **TextMLP + Boost-DD** workflow so we can re-use the image-based framework on the AG News text classification task. The steps mirror Tim’s Phase 0–4 pipeline.

## Environment setup
1. Create a fresh Conda environment (Python ≥ 3.10):
   ```bash
   conda create -n textdd python=3.10
   conda activate textdd
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   This covers PyTorch, `higher`, HuggingFace tooling, etc. When running on an HPC with special CUDA builds, swap the `torch/torchvision` wheels as required by the cluster.

## 0. Prepare sentence embeddings
1. Install optional dependencies: `pip install datasets transformers tqdm`.
2. Run the embedding script (defaults to `bert-base-uncased`):
   ```bash
   python scripts/prepare_agnews_embeddings.py --batch_size 64 --device cuda
   ```
   This produces `agnews_train_emb.pt`, `agnews_val_emb.pt`, etc. under `scripts/agnews_emb/`, which `main.py` reads automatically.

## 1–3. Text distillation with TextMLP + Boost-DD
- Launch distillation via `main.py`; defaults already point to `agnews_emb` and `text_mlp`. Pick your IPC, filenames, and stages.
- Example (stage 0, IPC=10):
  ```bash
  python main.py \
    --dataset agnews_emb \
    --arch text_mlp \
    --num_per_class 10 \
    --batch_per_class 5 \
    --task_sampler_nc 4 \
    --window 40 \
    --totwindow 120 \
    --epochs 2000 \
    --fname agnews_mlp_ipc10_s0 \
    --name agnews_step3_s0
  ```
- For subsequent Boost-DD stages append e.g. `--boost_dd --boost_init_from out_step3_agnews_ipc10_s0.h5 --boost_beta 0.3 --stage 1`, so synthetic blocks are warm-started.
- Outputs are HDF5/PyTorch files in the working directory (`out_step3_agnews_ipc10_s1.h5`, etc.).

## 4. Evaluate distilled sets
- **Distilled-set training:**  
  ```bash
  python eval_step4_agnews.py --data_root scripts/agnews_emb --paths out_step3_agnews_ipc10_s1.h5
  ```
  You can pass multiple `--paths`; each one trains `TextMLP` with 5 seeds and reports mean/std accuracy.
- **Random-subset baseline:**  
  ```bash
  python eval_baseline_random_agnews.py \
      --data_root scripts/agnews_emb \
      --ipcs 5 10 15 20 \
      --seeds 0 1 2 3 4
  ```
  This samples real data with the same IPC budget to compare against the distilled sets.

## Layout
```
haitong_code/
├── main.py
├── framework/
│   └── ... (base.py, config.py, text_nlp.py, etc.)
├── scripts/
│   └── prepare_agnews_embeddings.py
├── eval_step4_agnews.py
├── eval_baseline_random_agnews.py
├── requirements.txt
└── README.md
```

`framework/config.py` already includes the `agnews_emb` entry while keeping `mrpc_emb` support. To add another dataset, replicate the embedding script and config branch with new filenames.
