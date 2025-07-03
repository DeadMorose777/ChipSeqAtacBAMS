"""
Генерация train/val/test DataLoader-ов с единым collate_fn.
"""

import json, random
from pathlib import Path
from typing import Callable
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class JsonlDataset(Dataset):
    def __init__(self, path: Path, transform: Callable):
        self.recs = [json.loads(l) for l in open(path)]
        self.transform = transform

    def __len__(self): return len(self.recs)

    def __getitem__(self, idx):
        rec = self.recs[idx]
        return self.transform(rec)

def default_transform(rec: dict) -> dict:
    return {
        "seq"  : rec["seq"],
        "atac" : torch.tensor([rec["atac"]], dtype=torch.float32),
        "label": torch.tensor(rec["label"], dtype=torch.float32),
    }

def split_dataset(ds: Dataset, train_f: float, val_f: float, seed: int):
    n = len(ds)
    n_train = int(n * train_f)
    n_val   = int(n * val_f)
    torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val, n - n_train - n_val])

def build_loaders(cfg: dict, collate_fn) -> tuple[DataLoader, DataLoader, DataLoader]:
    ds = JsonlDataset(Path(cfg["dataset"]["path"]), default_transform)
    tr, va, te = split_dataset(ds,
                               cfg["dataset"]["train_frac"],
                               cfg["dataset"]["val_frac"],
                               cfg["random_seed"])
    kwargs = dict(batch_size=cfg["loader"]["batch_size"],
                  num_workers=cfg["loader"]["num_workers"],
                  collate_fn=collate_fn,
                  shuffle=True)
    return (DataLoader(tr, **kwargs),
            DataLoader(va, **kwargs),
            DataLoader(te, **kwargs))
