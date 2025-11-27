# scripts/prepare_mrpc_embeddings.py

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

def get_mrpc_embeddings(
    model_name="bert-base-uncased",
    split="train",
    batch_size=32,
    device="cuda"
):
    # 1) load MRPC
    ds = load_dataset("glue", "mrpc", split=split)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_embs = []
    all_labels = []

    for i in tqdm(range(0, len(ds), batch_size), desc=f"MRPC {split}"):
        batch = ds[i : i + batch_size]

        s1_list = batch["sentence1"]
        s2_list = batch["sentence2"]
        labels  = batch["label"]

        # [CLS] s1 [SEP] s2 [SEP]
        enc = tokenizer(
            s1_list,
            s2_list,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )

        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            # CLS embedding: [batch_size, hidden_size]
            cls_emb = outputs.last_hidden_state[:, 0, :]

        all_embs.append(cls_emb.cpu())
        all_labels.append(torch.tensor(labels, dtype=torch.long))

    embs = torch.cat(all_embs, dim=0)     # [N, d_emb]
    labels = torch.cat(all_labels, dim=0) # [N]

    return embs, labels


def main():
    # Where to save inside the repo
    out_dir = os.path.join("data", "MRPC_EMB")
    os.makedirs(out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = "bert-base-uncased"

    # Train split
    train_embs, train_labels = get_mrpc_embeddings(
        model_name=model_name,
        split="train",
        batch_size=32,
        device=device,
    )
    torch.save(train_embs,  os.path.join(out_dir, "mrpc_train_emb.pt"))
    torch.save(train_labels, os.path.join(out_dir, "mrpc_train_labels.pt"))

    # Validation split (GLUE MRPC has 'validation' not 'test')
    val_embs, val_labels = get_mrpc_embeddings(
        model_name=model_name,
        split="validation",
        batch_size=32,
        device=device,
    )
    torch.save(val_embs,  os.path.join(out_dir, "mrpc_val_emb.pt"))
    torch.save(val_labels, os.path.join(out_dir, "mrpc_val_labels.pt"))

    print("Done. Shapes:")
    print("train:", train_embs.shape, train_labels.shape)
    print("val:",   val_embs.shape,   val_labels.shape)


if __name__ == "__main__":
    main()
