# framework/text_nlp.py

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingTensorDataset(Dataset):
    """Loads precomputed embeddings + labels from .pt files."""
    def __init__(self, emb_path, label_path):
        super().__init__()
        self.x = torch.load(emb_path, map_location="cpu")  # [N, d_emb]
        self.y = torch.load(label_path, map_location="cpu").long()  # [N]

        assert self.x.ndim == 2, f"Expected [N, d_emb], got {self.x.shape}"
        assert self.y.ndim == 1, f"Expected [N], got {self.y.shape}"
        assert self.x.shape[0] == self.y.shape[0], "x and y size mismatch"

        self.N, self.d_emb = self.x.shape

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class TextMLP(nn.Module):
    def __init__(self, d_in, d_hidden, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


class TextTransformer(nn.Module):
    def __init__(
        self,
        d_in: int,
        num_classes: int,
        d_model: int = 256,
        nhead: int = 4,
        num_layers: int = 2,
        seq_len: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_in % seq_len != 0:
            raise ValueError(f"TextTransformer: d_in={d_in} must be divisible by seq_len={seq_len}")

        self.d_in = d_in
        self.d_model = d_model
        self.seq_len = seq_len

        token_dim = d_in // seq_len
        self.input_proj = nn.Linear(token_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls_head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        B, D = x.shape
        if D != self.d_in:
            raise ValueError(f"TextTransformer: expected d_in={self.d_in}, got {D}")
        if D % self.seq_len != 0:
            raise ValueError(f"TextTransformer: d_in={D} not divisible by seq_len={self.seq_len}")

        token_dim = D // self.seq_len
        x = x.view(B, self.seq_len, token_dim)
        x = self.input_proj(x)
        h = self.encoder(x)
        h = h.mean(dim=1)
        logits = self.cls_head(h)
        return logits
