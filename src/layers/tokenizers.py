"""
Разные способы превратить последовательность нуклеотидов в тензор.
"""

import numpy as np
import torch
from transformers import AutoTokenizer

class OneHotEncoder:
    _map = {b"A": 0, b"C": 1, b"G": 2, b"T": 3}

    def __init__(self, pad_to: int | None = None):
        self.pad_to = pad_to

    def __call__(self, seq: str) -> torch.Tensor:
        if self.pad_to and len(seq) < self.pad_to:
            seq = seq + "N" * (self.pad_to - len(seq))
        arr = np.zeros((4, len(seq)), dtype=np.float32)
        for i, b in enumerate(seq.encode()):
            if b in self._map:
                arr[self._map[b], i] = 1
        return torch.from_numpy(arr)

class DNATokenizer:
    """
    Обёртка над pretrained-tokenizer`ом (DNABERT-6, GenALM…).
    """
    def __init__(self, model_name: str, max_len: int = 256):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def __call__(self, seq: str) -> dict[str, torch.Tensor]:
        enc = self.tok(
            seq,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )
        return {k: v.squeeze(0) for k, v in enc.items()}
