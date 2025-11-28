import argparse
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from framework.config import get_arch  # use same arch factory as training
import h5py


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
    Load synthetic embeddings from either:
      - a PyTorch checkpoint (.pt/.pth) containing 'model_state_dict' or 'weight', OR
      - an HDF5 file (.h5/.hdf5) with dataset 'data'.

    Returns:
      syn_X: [N_syn, d_emb]
      syn_y: [N_syn]
      d_emb: int
    """
    # 1) Try as a PyTorch checkpoint first
    state = None
    is_torch_checkpoint = False
    try:
        state = torch.load(ckpt_path, map_location="cpu")
        is_torch_checkpoint = True
        print(f"[load_synthetic_data] Loaded '{ckpt_path}' as torch checkpoint.")
    except Exception as e:
        print(f"[load_synthetic_data] torch.load failed on '{ckpt_path}', "
              f"will try HDF5. Error: {e}")

    syn_X = None

    if is_torch_checkpoint:
        # Case 1: tensor directly
        if torch.is_tensor(state):
            syn_X = state.float()

        # Case 2: direct 'weight' key
        elif isinstance(state, dict) and "weight" in state and torch.is_tensor(state["weight"]):
            syn_X = state["weight"].float()

        # Case 3: our older style: state['model_state_dict']['module.data.weight']
        elif isinstance(state, dict) and "model_state_dict" in state:
            msd = state["model_state_dict"]
            if "module.data.weight" in msd:
                syn_X = msd["module.data.weight"].float()
            else:
                candidates = [k for k in msd.keys() if "data.weight" in k]
                if not candidates:
                    raise ValueError(
                        f"model_state_dict found in '{ckpt_path}', "
                        "but no 'module.data.weight' or similar key present."
                    )
                syn_X = msd[candidates[0]].float()
        else:
            # last resort: try any tensor in the dict
            if isinstance(state, dict):
                for v in state.values():
                    if torch.is_tensor(v):
                        syn_X = v.float()
                        break

        if syn_X is None:
            raise ValueError(
                f"Could not extract synthetic tensor from torch checkpoint '{ckpt_path}'."
            )

    else:
        # 2) Fallback: treat as HDF5
        print(f"[load_synthetic_data] Trying to load '{ckpt_path}' as HDF5...")
        with h5py.File(ckpt_path, "r") as f:
            if "data" in f:
                data_np = f["data"][:]
            else:
                # If no 'data' key, grab the first dataset we find
                keys = [k for k in f.keys()]
                if not keys:
                    raise ValueError(f"HDF5 file '{ckpt_path}' has no datasets.")
                print(f"[load_synthetic_data] HDF5 keys={keys}, using '{keys[0]}' as data.")
                data_np = f[keys[0]][:]
        syn_X = torch.tensor(data_np, dtype=torch.float32)

    # Now we have syn_X as [N_syn, d_emb]
    num_syn, d_emb = syn_X.shape
    expected_syn = num_classes * ipc
    if num_syn != expected_syn:
        print(f"[WARN] Synthetic data has {num_syn} rows, but num_classes*ipc={expected_syn}. "
              "This is fine if you're using different ipc; just make sure --ipc matches.")

    # Labels: [0..0, 1..1, ..., num_classes-1..]
    syn_y = torch.arange(num_classes).repeat_interleave(ipc)[:num_syn]

    return syn_X, syn_y, d_emb



def train_on_syn_and_eval(
    syn_X, syn_y, x_val, y_val,
    d_emb, num_classes, arch, hidden_dim,
    lr, epochs, batch_size, device
):
    """
    Train a fresh student model (TextMLP or TextTransformer) on the synthetic set,
    then evaluate on real MRPC val.
    """

    syn_X = syn_X.to(device)
    syn_y = syn_y.to(device)
    x_val = x_val.to(device)
    y_val = y_val.to(device)

    # For MRPC embeddings, treat them as [C, H, W] = [1, 1, d_emb]
    channel = 1
    im_size = (1, d_emb)

    print(f"Building student model arch={arch}, width={hidden_dim}, "
          f"d_emb={d_emb}, num_classes={num_classes}")

    model = get_arch(
        arch=arch,
        num_classes=num_classes,
        channel=channel,
        im_size=im_size,
        width=hidden_dim,  # for text_mlp: hidden size; for text_transformer: d_model
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    n_syn = syn_X.size(0)

    for epoch in range(epochs):
        # simple SGD/Adam over synthetic set
        permutation = torch.randperm(n_syn, device=device)
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
                        help="Path to synthetic checkpoint .pth/.h5 file")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes in MRPC (default 2)")
    parser.add_argument("--ipc", type=int, default=5,
                        help="Images (synthetic samples) per class")

    parser.add_argument("--arch", type=str, default="text_mlp",
                        help="Student architecture: 'text_mlp' or 'text_transformer'")

    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Width: hidden dim for TextMLP or d_model for TextTransformer")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate for student")
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
    syn_X, syn_y, d_emb = load_synthetic_data(
        args.syn_ckpt,
        num_classes=args.num_classes,
        ipc=args.ipc,
    )
    print(f"Synthetic set shape: X={tuple(syn_X.shape)}, y={tuple(syn_y.shape)}")

    # Try to force SDPA math backend (helpful if arch=text_transformer and using nn.Transformer)
    ctx = None
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        ctx = sdpa_kernel(SDPBackend.MATH)
        print("Using SDPA MATH backend for student training.")
    except Exception as e:
        from contextlib import nullcontext
        ctx = nullcontext()
        print(f"Could not configure SDPA kernel for eval (falling back to default): {e}")

    with ctx:
        print(f"Training fresh {args.arch} on synthetic set and evaluating on real MRPC val...")
        acc = train_on_syn_and_eval(
            syn_X, syn_y, x_val, y_val,
            d_emb=d_emb,
            num_classes=args.num_classes,
            arch=args.arch,
            hidden_dim=args.hidden_dim,
            lr=args.lr,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )

    print(f"\n=== FINAL DISTILLED-SET ACCURACY ON MRPC VAL: {acc*100:.2f}% ===")


if __name__ == "__main__":
    main()
