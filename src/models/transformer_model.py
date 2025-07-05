"""
TransformerModel 2.0
====================
⟶ Никаких HuggingFace-весов: всё создаётся с нуля в PyTorch.

Вход  :  • ДНК-строка длиной L  (символы A,C,G,T,N)
         • ATAC-вектор (1,L)

Архитектура:
    embed = TokEmbedding(id→d) + PosEmbedding +  Linear(ATAC→d)
    x     = TransformerEncoder(N layers, d_model, n_heads)
    logits= Linear(d_model→1)  → (B,L)

Конфиг (пример configs/transformer.yaml):
    d_model    : 128
    n_heads    : 8
    n_layers   : 4
    dropout    : 0.1
    seq_length : 1000
"""

from typing import List
import torch, torch.nn as nn
from .base_model import BaseModel

# ------------------------- helpers --------------------------------------- #
_DNA2ID = {b"A": 0, b"C": 1, b"G": 2, b"T": 3, b"N": 4}
VOCAB_SIZE = len(_DNA2ID)           # 5

def encode_seq(seq: str) -> torch.LongTensor:
    """str → LongTensor(L) с id букв"""
    arr = [ _DNA2ID.get(b, 4) for b in seq.encode() ]
    return torch.tensor(arr, dtype=torch.long)


class PositionalEncoding(nn.Module):
    """классический sinusoidal PE (fixed)"""
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) *
                        (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1,max_len,d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]           # broadcast batch


# ------------------------- main model ------------------------------------ #
class TransformerModel(BaseModel):
    def __init__(self, cfg: dict):
        super().__init__()
        d_model = cfg.get("d_model", 128)
        n_heads = cfg.get("n_heads", 8)
        n_layers= cfg.get("n_layers", 4)
        dropout = cfg.get("dropout", 0.1)
        max_len = cfg.get("seq_length", 1000)

        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
        self.atac_proj = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, 1)     # → логит

    # ---------- API ----------
    def forward(self, batch):
        ids   = batch["ids"]          # (B,L)
        atac  = batch["atac"].transpose(1,2)  # (B,L,1)

        x = self.tok_emb(ids) + self.atac_proj(atac)
        x = self.pos_emb(x)
        x = self.encoder(x)                    # (B,L,d)
        logits = self.head(x).squeeze(-1)      # (B,L)
        return logits

    def collate_fn(self, samples: List[dict]):
        ids   = torch.stack([encode_seq(s["seq"]) for s in samples])   # (B,L)
        atac  = torch.stack([s["atac"]  for s in samples])             # (B,1,L)
        label = torch.stack([s["label"] for s in samples])             # (B,L)
        return {"ids": ids, "atac": atac, "label": label}
