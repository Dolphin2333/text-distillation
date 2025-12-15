# eval_step5_overview.py

import argparse
import glob
import os
import re
import random

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.config import get_arch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_val_embeddings(data_root: str, emb_file: str, label_file: str):
    x_val = torch.load(os.path.join(data_root, emb_file), map_location="cpu").float()
    y_val = torch.load(os.path.join(data_root, label_file), map_location="cpu").long()
    return x_val, y_val


def load_synthetic_h5(path: str):
    """Load synthetic embeddings from a Step-5 HDF5 file with dataset 'data'."""
    with h5py.File(path, "r") as f:
        if "data" in f:
            data_np = f["data"][:]
        else:
            keys = list(f.keys())
            if not keys:
                raise ValueError(f"HDF5 file '{path}' has no datasets.")
            data_np = f[keys[0]][:]
    syn_X = torch.tensor(data_np, dtype=torch.float32)
    return syn_X


def parse_ipc_and_stage_from_name(path: str):
    """
    Tries to parse IPC and stage from filenames like:
      out_step5_ipc30_s5_tf_adamlr.h5
      out_step5_debug_ipc10_beta0.h5
    Fallback is None if pattern not found.
    """
    fname = os.path.basename(path)
    # search for 'ipc<digits>'
    m_ipc = re.search(r"ipc(\d+)", fname)
    ipc = int(m_ipc.group(1)) if m_ipc else None

    # search for '_s<digits>' as stage
    m_stage = re.search(r"_s(\d+)", fname)
    stage = int(m_stage.group(1)) if m_stage else None

    return ipc, stage


def train_student_on_syn(
    syn_X: torch.Tensor,
    syn_y: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    arch: str,
    hidden_dim: int,
    lr: float,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
    num_classes: int,
):
    set_seed(seed)

    syn_X = syn_X.to(device)
    syn_y = syn_y.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    if num_classes <= 0:
        num_classes = int(syn_y.max().item()) + 1
    d_emb = syn_X.size(1)

    channel = 1
    im_size = (1, d_emb)

    model = get_arch(
        arch=arch,
        num_classes=num_classes,
        channel=channel,
        im_size=im_size,
        width=hidden_dim,  # hidden size (MLP) or d_model (transformer)
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    n_syn = syn_X.size(0)

    for epoch in range(epochs):
        perm = torch.randperm(n_syn, device=device)
        for i in range(0, n_syn, batch_size):
            idx = perm[i : i + batch_size]
            xb = syn_X[idx]
            yb = syn_y[idx]

            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits_val = model(x_val)
        preds = logits_val.argmax(dim=1)
        acc = (preds == y_val).float().mean().item()

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="./scripts/mrpc_emb",
        help="Folder containing *_val_emb.pt files",
    )
    parser.add_argument("--val_emb_file", type=str, default="mrpc_val_emb.pt")
    parser.add_argument("--val_label_file", type=str, default="mrpc_val_labels.pt")
    parser.add_argument(
        "--paths",
        nargs="+",
        default=["out_step5_*.h5"],
        help="Relative paths for synthetic H5 files (Step 5).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Seeds for student training.",
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="text_transformer",
        help="Student arch: 'text_mlp' or 'text_transformer'.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
        help="Hidden size (MLP) or d_model (transformer).",
    )
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes in the dataset (e.g., 4 for AG News)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_cuda", action="store_true")

    args = parser.parse_args()

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    x_val, y_val = load_val_embeddings(args.data_root, args.val_emb_file, args.val_label_file)
    print(f"Validation set: x_val={tuple(x_val.shape)}, y_val={tuple(y_val.shape)}")

    # gather files
    paths = []
    for pat in args.paths:
        paths.extend(sorted(glob.glob(pat)))
    paths = sorted(set(paths))

    print(f"Found {len(paths)} synthetic files.")
    if not paths:
        return

    results = []

    for path in paths:
        ipc, stage = parse_ipc_and_stage_from_name(path)
        print(f"\n=== Evaluating file: {path} (ipc={ipc}, stage={stage}) ===")

        syn_X = load_synthetic_h5(path)
        if ipc is None:
            num_syn = syn_X.size(0)
            ipc = num_syn // args.num_classes
            print(
                f"  [WARN] IPC not parsed from name; inferred ipc={ipc} "
                f"from N={num_syn} and num_classes={args.num_classes}."
            )

        # build synthetic labels assuming [class0 block, class1 block, ...]
        syn_y = torch.arange(args.num_classes).repeat_interleave(ipc)[: syn_X.size(0)]

        accs = []
        for seed in args.seeds:
            print(f"  seed={seed} ...")
            acc = train_student_on_syn(
                syn_X,
                syn_y,
                x_val,
                y_val,
                arch=args.arch,
                hidden_dim=args.hidden_dim,
                lr=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                seed=seed,
                num_classes=args.num_classes,
            )
            accs.append(acc)
            print(f"    -> acc={acc:.4f}")

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        results.append(
            {
                "path": path,
                "ipc": ipc,
                "stage": stage,
                "mean_acc": mean_acc,
                "std_acc": std_acc,
            }
        )
        print(f"  [SUMMARY] mean={mean_acc:.4f}  std={std_acc:.4f}")

    # pretty table
    print("\n=== Summary over files ===")
    print("{:<40s} {:>6s} {:>6s} {:>12s} {:>12s}".format("file", "IPC", "stage", "mean_acc", "std_acc"))
    for r in sorted(results, key=lambda x: (x["ipc"] or 0, x["stage"] or 0)):
        print(
            "{:<40s} {:>6} {:>6} {:>11.4f} {:>11.4f}".format(
                os.path.basename(r["path"]),
                r["ipc"] if r["ipc"] is not None else "-",
                r["stage"] if r["stage"] is not None else "-",
                r["mean_acc"],
                r["std_acc"],
            )
        )


if __name__ == "__main__":
    main()
