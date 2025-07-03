"""
CNN из старого train_cnn.py :contentReference[oaicite:2]{index=2}, переписанный под общий интерфейс.
"""

import torch, torch.nn as nn
from .base_model import BaseModel
from ..layers import OneHotEncoder

class CNNModel(BaseModel):
    def __init__(self, cfg: dict):
        super().__init__()
        self.enc = OneHotEncoder(pad_to=cfg["seq_length"])
        self.conv = nn.Conv1d(4, cfg["n_filters"], cfg["kernel"])
        self.pool = nn.MaxPool1d(cfg["pool"])
        L = (cfg["seq_length"] - cfg["kernel"] + 1) // cfg["pool"]
        self.fc1 = nn.Linear(cfg["n_filters"] * L + 1, cfg["hidden"])
        self.fc2 = nn.Linear(cfg["hidden"], 1)

    # ---------- API ----------
    def forward(self, batch):
        x = batch["seq"]           # (B,4,L)
        at = batch["atac"]         # (B,1)
        x = self.pool(torch.relu(self.conv(x)))
        x = torch.cat([x.flatten(1), at], 1)
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x)).squeeze(1)

    def collate_fn(self, samples):
        seqs = torch.stack([self.enc(s["seq"]) for s in samples])
        atac = torch.stack([s["atac"] for s in samples])
        labels = torch.stack([s["label"] for s in samples])
        return {"seq": seqs, "atac": atac, "label": labels}
