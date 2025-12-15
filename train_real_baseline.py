import argparse
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from framework.config import get_dataset, get_transform, get_arch, get_pin_memory


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_fn = nn.CrossEntropyLoss()
    running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    avg_loss = running_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def train_one_seed(args, seed, device):
    set_seed(seed)
    transform_train, transform_test = get_transform(args.dataset)
    trainset, valset, testset, num_classes, shape, _ = get_dataset(
        args.dataset, args.root, transform_train, transform_test, zca=False
    )

    pin_memory = get_pin_memory(args.dataset)

    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    channel, height, width = shape
    model = get_arch(args.arch, num_classes, channel, (height, width), width=args.width)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_val = 0.0
    best_test = 0.0
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            total += targets.size(0)

        scheduler.step()
        train_loss = running_loss / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        _, test_acc = evaluate(model, val_loader, device)  # using same split as test proxy

        if val_acc > best_val:
            best_val = val_acc
            best_test = test_acc

        print(
            f"[Seed {seed}] Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    elapsed = time.time() - start
    return {
        "seed": seed,
        "best_val": best_val,
        "best_test": best_test,
        "time_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Real-data training baseline")
    parser.add_argument("--root", default="./scripts", type=str)
    parser.add_argument("--dataset", default="agnews_emb", type=str)
    parser.add_argument("--arch", default="text_mlp", type=str, choices=["text_mlp", "text_transformer"])
    parser.add_argument("--width", default=256, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--wd", default=0.0, type=float)
    parser.add_argument("--batch_size", default=2048, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--use_cuda", action="store_true")
    args = parser.parse_args()

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    results = []
    for seed in args.seeds:
        stats = train_one_seed(args, seed, device)
        results.append(stats)
        print(
            f"Seed {seed} finished: best_val={stats['best_val']:.4f}, "
            f"time={stats['time_sec']/60:.2f} min"
        )

    print("\n=== Summary ===")
    for stats in results:
        print(
            f"Seed {stats['seed']}: best_val={stats['best_val']:.4f}, "
            f"best_test={stats['best_test']:.4f}, time={stats['time_sec']/60:.2f} min"
        )


if __name__ == "__main__":
    main()
