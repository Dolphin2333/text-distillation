"""Prepare AG News sentence embeddings for text distillation.

This script mirrors the MRPC pipeline but targets the AG News dataset:
  * Loads AG News from HuggingFace datasets.
  * Encodes each article text using a frozen Transformer (default: bert-base-uncased).
  * Saves train/test (used as validation) embeddings/labels as .pt tensors.

Run:
    python scripts/prepare_agnews_embeddings.py \
        --model bert-base-uncased \
        --batch_size 64 \
        --device cuda
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm


def encode_split(dataset_name: str, split: str, tokenizer, model, batch_size: int, device: str):
    ds = load_dataset(dataset_name, split=split)
    embs, labels = [], []

    for idx in tqdm(range(0, len(ds), batch_size), desc=f"{split} batches"):
        batch = ds[idx: idx + batch_size]
        texts = batch["text"]
        labels.extend(batch["label"])

        enc = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            cls = outputs.last_hidden_state[:, 0, :]
            embs.append(cls.cpu())

    return torch.cat(embs, dim=0), torch.tensor(labels, dtype=torch.long)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="ag_news", help="HF dataset name")
    parser.add_argument("--model", default="bert-base-uncased", help="HF model name")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--out_dir",
        default=os.path.join("scripts", "agnews_emb"),
        help="Where to store *.pt tensors",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(args.device)
    model.eval()

    print(f"Encoding {args.dataset} train split ...")
    train_emb, train_lbl = encode_split(args.dataset, "train", tokenizer, model, args.batch_size, args.device)
    torch.save(train_emb, os.path.join(args.out_dir, "agnews_train_emb.pt"))
    torch.save(train_lbl, os.path.join(args.out_dir, "agnews_train_labels.pt"))
    print(f"Train tensors saved: {tuple(train_emb.shape)}, {tuple(train_lbl.shape)}")

    print(f"Encoding {args.dataset} test split (used as validation) ...")
    val_emb, val_lbl = encode_split(args.dataset, "test", tokenizer, model, args.batch_size, args.device)
    torch.save(val_emb, os.path.join(args.out_dir, "agnews_val_emb.pt"))
    torch.save(val_lbl, os.path.join(args.out_dir, "agnews_val_labels.pt"))
    print(f"Val tensors saved: {tuple(val_emb.shape)}, {tuple(val_lbl.shape)}")


if __name__ == "__main__":
    main()
