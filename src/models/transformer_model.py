"""
Файн-тюнинг DNABERT-6, адаптирован с train_transformer.py :contentReference[oaicite:3]{index=3}.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from .base_model import BaseModel
from ..layers import DNATokenizer

class TransformerModel(BaseModel):
    def __init__(self, cfg: dict):
        super().__init__()
        self.tok = DNATokenizer(cfg["model_name"])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg["model_name"], num_labels=1, problem_type="regression")
        if cfg.get("freeze_backbone", True):
            for p in self.model.base_model.parameters():
                p.requires_grad = False

    # ---------- API ----------
    def forward(self, batch):
        out = self.model(**batch["tok"])      # logits shape (B,1)
        return torch.sigmoid(out.logits.squeeze(1))

    def collate_fn(self, samples):
        toks = self.tok.tok.pad(
            [self.tok(s["seq"]) for s in samples],
            padding="longest",
            return_tensors="pt",
        )
        labels = torch.stack([s["label"] for s in samples])
        return {"tok": toks, "label": labels}
