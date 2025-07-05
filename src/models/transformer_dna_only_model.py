import torch
from .transformer_model import TransformerModel, encode_seq  # ← берём функцию индексации

class TransformerDNAOnlyModel(TransformerModel):
    """
    Transformer, который использует ТОЛЬКО последовательность ДНК.
    ATAC-канал заполняется нулями.
    """
    def __init__(self, cfg):
        super().__init__(cfg)   # вся архитектура та же

    # ---------- collate_fn ----------
    def collate_fn(self, samples):
        # ids: (B,L) целые индексы 0..4
        ids   = torch.stack([encode_seq(s["seq"]) for s in samples])
        L     = ids.shape[1]
        # пустой ATAC-канал (B,1,L)
        atac  = torch.zeros(len(samples), 1, L)
        label = torch.stack([s["label"] for s in samples])
        return {"ids": ids, "atac": atac, "label": label}
