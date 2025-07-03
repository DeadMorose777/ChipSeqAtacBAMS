"""
Генерация train/val/test DataLoader-ов с единым collate_fn.
Если в jsonl нет поля "seq", последовательность берётся из fasta.
"""

import json, torch, pyfaidx
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split

# ---------- helpers ----------
_FA_CACHE = {}                     # отдельный Fasta-объект на каждый worker


def get_fasta(path: str | Path):
    """Ленивая и потокобезопасная инициализация pyfaidx.Fasta."""
    global _FA_CACHE
    if path not in _FA_CACHE:
        _FA_CACHE[path] = pyfaidx.Fasta(str(path))
    return _FA_CACHE[path]


def make_transform(cfg: dict):
    fasta_path = Path(cfg["dataset"]["fasta"])
    win        = int(cfg["dataset"]["window"])

    def _transform(rec: dict) -> dict:
        # --- последовательность ---
        if "seq" in rec:                       # вдруг уже есть
            seq = rec["seq"].upper()
        else:
            fa = get_fasta(fasta_path)
            start = int(rec["start"])
            end   = start + win
            seq = fa[rec["chrom"]][start:end].seq.upper()

        return {
            "seq": seq,                                       # строка
            "atac": torch.tensor([float(rec["atac"])], dtype=torch.float32),
            "label": torch.tensor(float(rec["label"]), dtype=torch.float32),
        }

    return _transform


# ---------- Dataset / DataLoaders ----------
class JsonlDataset(Dataset):
    def __init__(self, path: Path, transform):
        self.recs = [json.loads(l) for l in open(path)]
        self.transform = transform

    def __len__(self): return len(self.recs)

    def __getitem__(self, idx):
        return self.transform(self.recs[idx])


def split_dataset(ds, train_f, val_f, seed):
    n = len(ds)
    n_train = int(n * train_f)
    n_val   = int(n * val_f)
    g = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_train, n_val, n - n_train - n_val], generator=g)


def build_loaders(cfg: dict, collate_fn):
    transform = make_transform(cfg)
    ds = JsonlDataset(Path(cfg["dataset"]["path"]), transform)

    tr, va, te = split_dataset(
        ds, cfg["dataset"]["train_frac"], cfg["dataset"]["val_frac"], cfg["random_seed"]
    )

    kwargs = dict(
        batch_size=cfg["loader"]["batch_size"],
        num_workers=cfg["loader"]["num_workers"],
        collate_fn=collate_fn,
        shuffle=True,
    )
    return DataLoader(tr, **kwargs), DataLoader(va, **kwargs), DataLoader(te, **kwargs)
