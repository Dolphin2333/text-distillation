# framework/text_nlp.py

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingTensorDataset(Dataset):
    """
    Simple dataset: loads precomputed embeddings and labels from .pt files.
    Each sample: (x, y) where
      x: [d_emb]
      y: scalar class index
    """
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


# framework/text_nlp.py

class TextMLP(nn.Module):
    """
    Tiny MLP for classification on embeddings.
    """
    def __init__(self, d_in, d_hidden, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc2 = nn.Linear(d_hidden, num_classes)

    def forward(self, x):
        # x can be [B, d_in] (real data) or [B, 1, H, W] (synthetic reshaped)
        x = x.view(x.size(0), -1)   # <-- ADD THIS LINE

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

