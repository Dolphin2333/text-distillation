# scripts/prepare_agnews_embeddings.py

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def get_agnews_embeddings(
    model_name="bert-base-uncased",
    split="train",
    batch_size=32,
    device="cuda",
):
    """Encode AG News samples into CLS embeddings."""
    ds = load_dataset("ag_news", split=split)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embs = []
    all_labels = []

    for i in tqdm(range(0, len(ds), batch_size), desc=f"AGNews {split}"):
        batch = ds[i : i + batch_size]
        texts = batch["text"]
        labels = batch["label"]

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
            cls_emb = outputs.last_hidden_state[:, 0, :]

        all_embs.append(cls_emb.cpu())
        all_labels.append(torch.tensor(labels, dtype=torch.long))

    embs = torch.cat(all_embs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    return embs, labels


def main():
    out_dir = os.path.join("scripts", "agnews_emb")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "bert-base-uncased"

    # Train split
    train_embs, train_labels = get_agnews_embeddings(
        model_name=model_name,
        split="train",
        batch_size=64,
        device=device,
    )
    torch.save(train_embs, os.path.join(out_dir, "agnews_train_emb.pt"))
    torch.save(train_labels, os.path.join(out_dir, "agnews_train_labels.pt"))

    # Test split (used as validation)
    val_embs, val_labels = get_agnews_embeddings(
        model_name=model_name,
        split="test",
        batch_size=64,
        device=device,
    )
    torch.save(val_embs, os.path.join(out_dir, "agnews_val_emb.pt"))
    torch.save(val_labels, os.path.join(out_dir, "agnews_val_labels.pt"))

    print("Done. Shapes:")
    print("train:", train_embs.shape, train_labels.shape)
    print("val:", val_embs.shape, val_labels.shape)


if __name__ == "__main__":
    main()
