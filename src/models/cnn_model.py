import torch, torch.nn as nn
from .base_model import BaseModel
from ..layers.tokenizers import OneHotFixed

class CNNModel(BaseModel):
    """5-канальный Conv-UNet-Lite: (DNA4+ATAC1) → logits(L)"""
    def __init__(self, cfg):
        super().__init__()
        L = cfg["seq_length"]
        self.enc = OneHotFixed()

        C = 128
        self.net = nn.Sequential(
            nn.Conv1d(5,C,7,padding=3), nn.ReLU(),
            nn.Conv1d(C,C,7,padding=3), nn.ReLU(),
            nn.Conv1d(C,C,7,padding=3), nn.ReLU(),
            nn.Conv1d(C,1,1)            # logits
        )

    # ---------- API ----------
    def forward(self, batch):
        x_dna  = batch["seq"]          # (B,4,L)
        x_atac = batch["atac"]         # (B,1,L)
        x = torch.cat([x_dna, x_atac],1)
        return self.net(x).squeeze(1)  # (B,L) logits

    def collate_fn(self, samples):
        dna = torch.stack([self.enc(s["seq"]) for s in samples])     # (B,4,L)
        atac= torch.stack([s["atac"] for s in samples])              # (B,1,L)
        lbl = torch.stack([s["label"] for s in samples])             # (B,L)
        return {"seq":dna,"atac":atac,"label":lbl}
