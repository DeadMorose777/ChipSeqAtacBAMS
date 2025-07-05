import torch
from .transformer_model import TransformerModel

class TransformerATACOnlyModel(TransformerModel):
    """
    Transformer, который видит ТОЛЬКО профиль ATAC (1,L),
    а ДНК-канал заполняет нулями.
    """
    def __init__(self, cfg):
        super().__init__(cfg)

    def collate_fn(self, samples):
        # DNA-пустышка = all zeros той же длины
        dummy_ids = torch.zeros(len(samples[0]["seq"]), dtype=torch.long)
        ids   = torch.stack([dummy_ids for _ in samples])            # (B,L)
        atac  = torch.stack([s["atac"] for s in samples])            # (B,1,L)
        label = torch.stack([s["label"] for s in samples])
        return {"ids": ids, "atac": atac, "label": label}
