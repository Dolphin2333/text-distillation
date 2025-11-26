import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from framework.text_nlp import TextMLP


def load_mrpc_data():
    X_train = torch.load("scripts/mrpc_emb/mrpc_train_emb.pt").float()
    y_train = torch.load("scripts/mrpc_emb/mrpc_train_labels.pt").long()

    X_val = torch.load("scripts/mrpc_emb/mrpc_val_emb.pt").float()
    y_val = torch.load("scripts/mrpc_emb/mrpc_val_labels.pt").long()

    return X_train, y_train, X_val, y_val


def sample_real_subset(X_train, y_train, ipc, num_classes=2, seed=0):
    """
    Take ipc examples per class from real MRPC training data.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    idxs = []
    for c in range(num_classes):
        class_idx = (y_train == c).nonzero(as_tuple=True)[0]
        # randomly choose ipc indices from this class
        perm = torch.randperm(class_idx.shape[0])[:ipc]
        idxs.append(class_idx[perm])
    idxs = torch.cat(idxs, dim=0)

    X_sub = X_train[idxs]       # (ipc * num_classes, 768)
    y_sub = y_train[idxs]       # (ipc * num_classes,)
    return X_sub, y_sub


def train_student(X_sub, y_sub, X_val, y_val, seed, epochs=200):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TextMLP(
        d_in=768,
        d_hidden=256,   # make sure matches your distillation MLP
        num_classes=2
    ).cuda()

    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    X_sub = X_sub.cuda()
    y_sub = y_sub.cuda()
    X_val = X_val.cuda()
    y_val = y_val.cuda()

    for _ in range(epochs):
        opt.zero_grad()
        logits = model(X_sub)
        loss = loss_fn(logits, y_sub)
        loss.backward()
        opt.step()

    with torch.no_grad():
        preds = model(X_val).argmax(dim=1)
        acc = (preds == y_val).float().mean().item()

    return acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ipcs", nargs="+", type=int, default=[5,10,15,20,25,30])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0,1,2,3,4])
    args = parser.parse_args()

    X_train, y_train, X_val, y_val = load_mrpc_data()

    print("\n===== RANDOM REAL-SUBSET BASELINE (MRPC) =====")
    print("IPCs:", args.ipcs)
    print("Seeds:", args.seeds)

    for ipc in args.ipcs:
        accs = []
        print(f"\n--- IPC = {ipc} ---")
        for seed in args.seeds:
            X_sub, y_sub = sample_real_subset(X_train, y_train, ipc, num_classes=2, seed=seed)
            acc = train_student(X_sub, y_sub, X_val, y_val, seed=seed, epochs=200)
            accs.append(acc)
            print(f"  seed={seed}: acc={acc:.4f}")
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        print(f"  -> MEAN={mean_acc:.4f}  STD={std_acc:.4f}")


if __name__ == "__main__":
    main()
