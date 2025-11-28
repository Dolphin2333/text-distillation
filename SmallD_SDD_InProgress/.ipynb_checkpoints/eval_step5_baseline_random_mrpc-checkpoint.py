# eval_step5_baseline_random_mrpc.py

import argparse
import os
import random

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


def load_mrpc_data(data_root: str):
    X_train = torch.load(os.path.join(data_root, "mrpc_train_emb.pt"), map_location="cpu").float()
    y_train = torch.load(os.path.join(data_root, "mrpc_train_labels.pt"), map_location="cpu").long()
    X_val = torch.load(os.path.join(data_root, "mrpc_val_emb.pt"), map_location="cpu").float()
    y_val = torch.load(os.path.join(data_root, "mrpc_val_labels.pt"), map_location="cpu").long()
    return X_train, y_train, X_val, y_val


def sample_real_subset(X_train, y_train, ipc: int, num_classes: int, seed: int):
    """
    Sample ipc real examples per class from the train set.
    """
    set_seed(seed)

    X_list = []
    y_list = []

    for c in range(num_classes):
        idx_c = (y_train == c).nonzero(as_tuple=True)[0]
        idx_c = idx_c[torch.randperm(len(idx_c))[:ipc]]
        X_list.append(X_train[idx_c])
        y_list.append(torch.full((ipc,), c, dtype=torch.long))

    X_sub = torch.cat(X_list, dim=0)
    y_sub = torch.cat(y_list, dim=0)
    return X_sub, y_sub


def train_student(
    X_sub,
    y_sub,
    X_val,
    y_val,
    arch: str,
    hidden_dim: int,
    lr: float,
    epochs: int,
    batch_size: int,
    device: str,
    seed: int,
):
    set_seed(seed)

    X_sub = X_sub.to(device)
    y_sub = y_sub.to(device)
    X_val = X_val.to(device)
    y_val = y_val.to(device)

    num_classes = int(y_sub.max().item()) + 1
    d_emb = X_sub.size(1)

    channel = 1
    im_size = (1, d_emb)

    model = get_arch(
        arch=arch,
        num_classes=num_classes,
        channel=channel,
        im_size=im_size,
        width=hidden_dim,
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    n_sub = X_sub.size(0)

    for epoch in range(epochs):
        perm = torch.randperm(n_sub, device=device)
        for i in range(0, n_sub, batch_size):
            idx = perm[i : i + batch_size]
            xb = X_sub[idx]
            yb = y_sub[idx]

            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        logits_val = model(X_val)
        preds = logits_val.argmax(dim=1)
        acc = (preds == y_val).float().mean().item()

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="./scripts/mrpc_emb",
        help="Folder with mrpc_*_emb.pt/mrpc_*_labels.pt",
    )
    parser.add_argument(
        "--ipcs",
        nargs="+",
        type=int,
        default=[5, 10, 15, 20, 25, 30],
        help="IPC values to test for baseline.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2],
        help="Seeds for random subsets.",
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
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--use_cuda", action="store_true")

    args = parser.parse_args()

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    X_train, y_train, X_val, y_val = load_mrpc_data(args.data_root)
    num_classes = int(y_train.max().item()) + 1
    print(f"Train: X={tuple(X_train.shape)}, y={tuple(y_train.shape)}; num_classes={num_classes}")
    print(f"Val:   X={tuple(X_val.shape)}, y={tuple(y_val.shape)}")

    for ipc in args.ipcs:
        print(f"\n--- IPC = {ipc} ---")
        accs = []
        for seed in args.seeds:
            X_sub, y_sub = sample_real_subset(X_train, y_train, ipc, num_classes=num_classes, seed=seed)
            acc = train_student(
                X_sub,
                y_sub,
                X_val,
                y_val,
                arch=args.arch,
                hidden_dim=args.hidden_dim,
                lr=args.lr,
                epochs=args.epochs,
                batch_size=args.batch_size,
                device=device,
                seed=seed,
            )
            accs.append(acc)
            print(f"  seed={seed}: acc={acc:.4f}")
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        print(f"  -> MEAN={mean_acc:.4f}  STD={std_acc:.4f}")


if __name__ == "__main__":
    main()
