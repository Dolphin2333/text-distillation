import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.text_nlp import TextMLP  # your MLP
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_val_data(data_root):
    """
    Load MRPC validation embeddings and labels.
    Assumes:
      data_root/mrpc_val_emb.pt      (tensor [N, d_emb])
      data_root/mrpc_val_labels.pt   (tensor [N])
    """
    emb_path = os.path.join(data_root, "mrpc_val_emb.pt")
    label_path = os.path.join(data_root, "mrpc_val_labels.pt")

    x_val = torch.load(emb_path, map_location="cpu")
    y_val = torch.load(label_path, map_location="cpu").long()

    return x_val, y_val


def load_synthetic_data(ckpt_path, num_classes, ipc):
    """
    Load synthetic embeddings from the .pth checkpoint.

    For your codebase, checkpoint looks like:
      {
        'epoch': ...,
        'model_state_dict': {
            'module.data.weight': tensor [num_classes*ipc, d_emb]
        },
        'optimizer_state_dict': ...,
        (maybe) 'label_state_dict': {...}
      }
    """
    state = torch.load(ckpt_path, map_location="cpu")

    # Case 1: they literally saved just the weight tensor (not your case, but future-proof)
    if torch.is_tensor(state):
        syn_X = state

    # Case 2: dict with a direct 'weight' key (other repos sometimes do this)
    elif isinstance(state, dict) and "weight" in state and torch.is_tensor(state["weight"]):
        syn_X = state["weight"]

    # Case 3: your actual case: 'model_state_dict' with 'module.data.weight'
    elif isinstance(state, dict) and "model_state_dict" in state:
        msd = state["model_state_dict"]
        # Try the exact key first
        if "module.data.weight" in msd:
            syn_X = msd["module.data.weight"]
        else:
            # Fallback: look for any key containing 'data.weight'
            candidates = [k for k in msd.keys() if "data.weight" in k]
            if not candidates:
                raise ValueError(
                    "model_state_dict found, but no 'module.data.weight' or similar key present."
                )
            syn_X = msd[candidates[0]]

    else:
        # Last-resort fallback: search through nested dicts / tensors
        syn_X = None
        if isinstance(state, dict):
            for v in state.values():
                # nested state_dict
                if isinstance(v, dict) and "module.data.weight" in v:
                    syn_X = v["module.data.weight"]
                    break
                if torch.is_tensor(v):
                    syn_X = v
                    break

        if syn_X is None:
            raise ValueError("Could not find synthetic weight tensor in checkpoint.")

    num_syn, d_emb = syn_X.shape
    expected_syn = num_classes * ipc
    if num_syn != expected_syn:
        print(f"[WARN] Synthetic data has {num_syn} rows, but "
              f"num_classes * ipc = {expected_syn}. "
              f"Will still proceed, but double-check this later.")

    # Integer labels [0,0,...,1,1,...] assuming class order 0..num_classes-1
    syn_y = torch.arange(num_classes).repeat_interleave(ipc)[:num_syn]

    return syn_X, syn_y, d_emb



def train_on_syn_and_eval(syn_X, syn_y, x_val, y_val,
                          d_emb, num_classes, hidden_dim,
                          lr, epochs, batch_size, device):

    syn_X = syn_X.to(device)
    syn_y = syn_y.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    # model = TextMLP(d_in=d_emb, d_hidden=hidden_dim, num_classes=num_classes).to(device)
    model = TextMLP(d_in=d_emb, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    n_syn = syn_X.size(0)

    for epoch in range(epochs):
        # simple SGD over synthetic set
        permutation = torch.randperm(n_syn)
        for i in range(0, n_syn, batch_size):
            idx = permutation[i:i + batch_size]
            batch_x = syn_X[idx]
            batch_y = syn_y[idx]

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % max(1, epochs // 5) == 0:
            # quick interim val accuracy
            model.eval()
            with torch.no_grad():
                val_logits = model(x_val)
                preds = val_logits.argmax(dim=1)
                acc = (preds == y_val).float().mean().item()
            print(f"[Epoch {epoch+1}/{epochs}] "
                  f"train_loss={loss.item():.4f} "
                  f"val_acc={acc*100:.2f}%")
            model.train()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        val_logits = model(x_val)
        preds = val_logits.argmax(dim=1)
        acc = (preds == y_val).float().mean().item()

    return acc


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, default="./scripts/mrpc_emb",
                        help="Path to MRPC embedding folder (mrpc_*_emb.pt, mrpc_*_labels.pt)")
    parser.add_argument("--syn_ckpt", type=str, required=True,
                        help="Path to synthetic checkpoint .pth file")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes in MRPC (default 2)")
    parser.add_argument("--ipc", type=int, default=5,
                        help="Images (synthetic samples) per class")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden dimension of TextMLP")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for student MLP")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="Number of epochs to train on synthetic set")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size over synthetic set")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed")
    parser.add_argument("--use_cuda", action="store_true",
                        help="Use CUDA if available")

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    print(f"Loading validation data from {args.data_root}...")
    x_val, y_val = load_val_data(args.data_root)
    print(f"Val set shape: x={tuple(x_val.shape)}, y={tuple(y_val.shape)}")

    print(f"Loading synthetic data from {args.syn_ckpt}...")
    syn_X, syn_y, d_emb = load_synthetic_data(args.syn_ckpt,
                                              num_classes=args.num_classes,
                                              ipc=args.ipc)
    print(f"Synthetic set shape: X={tuple(syn_X.shape)}, y={tuple(syn_y.shape)}")

    print("Training fresh TextMLP on synthetic set and evaluating on real MRPC val...")
    acc = train_on_syn_and_eval(
        syn_X, syn_y, x_val, y_val,
        d_emb=d_emb,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )

    print(f"\n=== FINAL DISTILLED-SET ACCURACY ON MRPC VAL: {acc*100:.2f}% ===")


if __name__ == "__main__":
    main()
