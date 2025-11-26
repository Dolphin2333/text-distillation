import os
import glob
import re
import h5py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Simple MLP student (same spirit as text_mlp)
# -----------------------------
class TextMLP(nn.Module):
    def __init__(self, d_in, d_hidden, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, num_classes)

    def forward(self, x):
        # x may be [B, d_in] or [B, 1, H, W]; flatten either way
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def load_mrpc_val(root="./scripts/mrpc_emb"):
    """Load MRPC validation embeddings and labels."""
    val_emb_path = os.path.join(root, "mrpc_val_emb.pt")
    val_label_path = os.path.join(root, "mrpc_val_labels.pt")

    val_emb = torch.load(val_emb_path)  # [N_val, d_emb]
    val_labels = torch.load(val_label_path).long()  # [N_val]

    return val_emb, val_labels


def load_synthetic_h5(path, num_classes=None):
    """
    Load synthetic dataset from an HDF5 file with 'data' and optional 'label'.

    If 'label' is missing, assume balanced classes:
        IPC = N / num_classes
        labels = sorted([0,1,...,num_classes-1] * IPC)
    """
    with h5py.File(path, "r") as f:
        if "data" not in f:
            raise KeyError(f"{path} does not contain dataset 'data'")
        data_np = f["data"][:]
        data = torch.tensor(data_np).float()

        labels = None
        if "label" in f:
            labels_np = f["label"][:]
            labels = torch.tensor(labels_np)
        else:
            if num_classes is None:
                raise ValueError(
                    f"{path} has no 'label' dataset and num_classes was not provided."
                )
            # Infer IPC from N and num_classes
            N = data.shape[0]
            if N % num_classes != 0:
                raise ValueError(
                    f"{path}: N={N} not divisible by num_classes={num_classes}; "
                    f"cannot infer balanced labels."
                )
            ipc = N // num_classes
            y_list = list(range(num_classes)) * ipc
            y_list.sort()
            labels = torch.tensor(y_list, dtype=torch.long)

    return data, labels


def train_student_on_synthetic(
    syn_x,
    syn_y,
    val_x,
    val_y,
    d_hidden=256,
    lr=1e-2,
    epochs=500,
    device="cuda",
):
    """
    Train a fresh TextMLP student on the synthetic dataset only,
    then evaluate on real MRPC validation.
    """
    num_classes = int(val_y.max().item() + 1)
    d_in = syn_x.shape[1]

    model = TextMLP(d_in=d_in, d_hidden=d_hidden, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    syn_x = syn_x.to(device)
    syn_y = syn_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(syn_x)
        loss = criterion(logits, syn_y)
        loss.backward()
        optimizer.step()

    # Eval on MRPC validation
    model.eval()
    with torch.no_grad():
        logits_val = model(val_x)
        pred = logits_val.argmax(dim=1)
        correct = (pred == val_y).sum().item()
        acc = correct / val_y.numel()

    return acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load MRPC validation set once
    val_x, val_y = load_mrpc_val(root="./scripts/mrpc_emb")
    num_classes = int(val_y.max().item() + 1)
    print(f"MRPC val: {val_x.shape[0]} samples, dim={val_x.shape[1]}, num_classes={num_classes}")

    # Find all Boost-DD synthetic .h5 outputs
    h5_paths = sorted(glob.glob("out_step3*_adamlr.h5"))
    if not h5_paths:
        print("No out_step2_mrpc_emb_text_mlp_ipc*_s*.h5 files found in current directory.")
        return

    print("\nFound synthetic checkpoints:")
    for p in h5_paths:
        print("  ", p)

    print("\nEvaluating each synthetic dataset with a fresh student (5 random restarts)...\n")

    results = []

    for path in h5_paths:
        # Try to parse IPC and stage from filename: ipc{IPC}_s{stage}
        basename = os.path.basename(path)
        m = re.search(r"ipc(\d+)_s(\d+)", basename)
        if m:
            ipc = int(m.group(1))
            stage = int(m.group(2))
        else:
            # fallback: unknown IPC/stage
            ipc = None
            stage = None

        print("=" * 80)
        print(f"FILE: {basename}")
        if ipc is not None:
            print(f"  Parsed IPC: {ipc}, stage: {stage}")
        print("-" * 80)

        syn_x, syn_y = load_synthetic_h5(path, num_classes=num_classes)
        N, d_emb = syn_x.shape
        print(f"  Synthetic data: N={N}, d_emb={d_emb}")
        if ipc is None:
            if N % num_classes == 0:
                ipc = N // num_classes
        print(f"  Inferred IPC: {ipc}")

        # Multiple random restarts
        n_repeats = 5
        accs = []
        for r in range(n_repeats):
            acc = train_student_on_synthetic(
                syn_x,
                syn_y,
                val_x,
                val_y,
                d_hidden=256,
                lr=1e-2,
                epochs=500,
                device=device,
            )
            accs.append(acc)
            print(f"    Run {r+1}/{n_repeats}: val_acc = {acc*100:.2f}%")

        accs_tensor = torch.tensor(accs)
        mean_acc = accs_tensor.mean().item()
        std_acc = accs_tensor.std(unbiased=False).item()
        print(f"  ==> Mean val accuracy over {n_repeats} runs: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%\n")

        results.append({
            "path": basename,
            "ipc": ipc,
            "stage": stage,
            "mean_acc": mean_acc,
            "std_acc": std_acc,
        })

    # Final summary table
    print("\n" + "#" * 80)
    print("FINAL SUMMARY (sorted by IPC)")
    print("#" * 80)

    # Sort by IPC then stage
    results_sorted = sorted(results, key=lambda d: (d["ipc"] if d["ipc"] is not None else math.inf,
                                                    d["stage"] if d["stage"] is not None else math.inf))

    print("{:<30s} {:>6s} {:>6s} {:>12s} {:>12s}".format("file", "IPC", "stage", "mean_acc", "std_acc"))
    for r in results_sorted:
        print("{:<30s} {:>6} {:>6} {:>11.4f} {:>11.4f}".format(
            r["path"],
            r["ipc"] if r["ipc"] is not None else "-",
            r["stage"] if r["stage"] is not None else "-",
            r["mean_acc"],
            r["std_acc"],
        ))


if __name__ == "__main__":
    main()
