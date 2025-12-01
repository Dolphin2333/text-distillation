import argparse
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

from framework.text_nlp import TextMLP


##########################################################
# Load real MRPC embedding data (YOUR ACTUAL FILES)
##########################################################
def load_agnews_data(root="scripts/agnews_emb"):
    X_train = torch.load(f"{root}/agnews_train_emb.pt").float()
    y_train = torch.load(f"{root}/agnews_train_labels.pt").long()

    X_val = torch.load(f"{root}/agnews_val_emb.pt").float()
    y_val = torch.load(f"{root}/agnews_val_labels.pt").long()

    return X_train, y_train, X_val, y_val


##########################################################
# Load synthetic distilled dataset
##########################################################
def load_synthetic(path, num_classes=4):
    with h5py.File(path, "r") as f:
        X = torch.tensor(f["data"][:]).float()      # (N, 768)

    N = X.shape[0]
    assert N % num_classes == 0, f"Total rows {N} not divisible by num_classes={num_classes}"
    ipc = N // num_classes

    # class 0 block, then class 1 block, etc.
    y = torch.arange(num_classes).repeat_interleave(ipc).long()

    return X, y



##########################################################
# Train student model on synthetic data
##########################################################
def train_student(X_syn, y_syn, X_val, y_val, seed, epochs=200, d_in=768):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TextMLP(
        d_in=d_in,
        d_hidden=256,
        num_classes=y_syn.max().item() + 1
    ).cuda()

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    X_syn = X_syn.cuda()
    y_syn = y_syn.cuda()
    X_val = X_val.cuda()
    y_val = y_val.cuda()

    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_syn)
        loss = loss_fn(logits, y_syn)
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = model(X_val).argmax(dim=1)
        acc = (preds == y_val).float().mean().item()

    return acc


##########################################################
# Main
##########################################################
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths", nargs="+", required=True)
    parser.add_argument("--data_root", default="scripts/agnews_emb")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4])
    args = parser.parse_args()

    X_train, y_train, X_val, y_val = load_agnews_data(args.data_root)
    d_in = X_train.size(1)

    print("\n======== STEP 4: FINAL EVALUATION ========")

    results = []

    for path in args.paths:
        print(f"\n--- Evaluating {path} ---")

        X_syn, y_syn = load_synthetic(path, num_classes=int(y_train.max().item()) + 1)
        accs = []

        for seed in args.seeds:
            acc = train_student(X_syn, y_syn, X_val, y_val, seed, d_in=d_in)
            accs.append(acc)
            print(f"  seed={seed}: acc={acc:.4f}")

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)

        print(f"  --> MEAN={mean_acc:.4f}  STD={std_acc:.4f}")

        results.append((path, mean_acc, std_acc))

    print("\n========== FINAL SUMMARY ==========")
    for p, m, s in results:
        print(f"{p:35s}   mean={m:.4f}   std={s:.4f}")
    print("===================================")


if __name__ == "__main__":
    main()
