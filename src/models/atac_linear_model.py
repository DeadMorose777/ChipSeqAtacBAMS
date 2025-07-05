"""
ATAC-Linear baseline
————————
logit_i = w * ATAC_i + b      (два обучаемых параметра на весь геном)
Не использует ДНК, поэтому «честный» baseline.
"""

import torch, torch.nn as nn
from .base_model import BaseModel

class ATACLinearModel(BaseModel):
    def __init__(self, cfg=None):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(0.0))
        self.b = nn.Parameter(torch.tensor(0.0))

    # ---------- API ----------
    def forward(self, batch):
        atac = batch["atac"].squeeze(1)          # (B,L)
        return self.w * atac + self.b            # (B,L) logits

    def collate_fn(self, samples):
        # из samples берём только ATAC и label
        atac  = torch.stack([s["atac"]  for s in samples])   # (B,1,L)
        label = torch.stack([s["label"] for s in samples])   # (B,L)
        return {"atac": atac, "label": label}
