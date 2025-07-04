"""
Dataset: каждая запись -> (seq_str, atac_vector(L), label_mask(L))
Кэшируем Fasta и BigWig, чтобы не открывать файл сотни раз.
"""

import json, torch, numpy as np, pyfaidx, pyBigWig
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split

_FASTA_CACHE  = {}
_BW_CACHE     = {}

def _open_fasta(path):
    if path not in _FASTA_CACHE:
        _FASTA_CACHE[path] = pyfaidx.Fasta(path)
    return _FASTA_CACHE[path]

def _open_bw(path):
    if path not in _BW_CACHE:
        _BW_CACHE[path] = pyBigWig.open(path)
    return _BW_CACHE[path]

def make_transform(cfg):
    win = cfg["dataset"]["window"]
    thr = cfg["dataset"]["fe_thresh"]

    def _tr(rec):
        fa   = _open_fasta(rec["dna_fa"])
        atac = _open_bw(rec["atac_bw"])
        fe   = _open_bw(rec["fe_bw"])

        s = int(rec["start"]); e = s+win
        chrom = rec["chrom"]

        # --- проверяем границы ещё раз ---
        chrom_len = min(atac.chroms().get(chrom,0),
                        fe.chroms().get(chrom,0))
        if e > chrom_len or s < 0:
            return None                     # будет отброшено DataLoader'ом

        seq = fa[chrom][s:e].seq.upper()
        atac_v = np.nan_to_num(atac.values(chrom,s,e), nan=0.0).astype("float32")
        fe_v   = np.nan_to_num(fe.values  (chrom,s,e), nan=0.0).astype("float32")
        label  = (fe_v > thr).astype("float32")

        return dict(seq=seq,
                    atac=torch.from_numpy(atac_v)[None,:],
                    label=torch.from_numpy(label))
    return _tr

class JsonlDataset(Dataset):
    def __init__(self, path: str, transform):
        self.recs = [json.loads(l) for l in open(path)]
        self.tr   = transform
    def __len__(self): return len(self.recs)
    def __getitem__(self,i):
        out = self.tr(self.recs[i])
        if out is None:                      # окно вышло за границы
            return self.__getitem__((i+1)%len(self))
        return out

def split_dataset(ds, train_f, val_f, seed):
    n = len(ds); n_tr = int(n*train_f); n_val = int(n*val_f)
    g = torch.Generator().manual_seed(seed)
    return random_split(ds, [n_tr, n_val, n-n_tr-n_val], generator=g)

def build_loaders(cfg, collate_fn):
    ds = JsonlDataset(cfg["dataset"]["path"], make_transform(cfg))
    tr,va,te = split_dataset(ds, cfg["dataset"]["train_frac"],
                                  cfg["dataset"]["val_frac"],
                                  cfg["random_seed"])
    kwargs = dict(batch_size=cfg["loader"]["batch_size"],
                  num_workers=cfg["loader"]["num_workers"],
                  collate_fn=collate_fn, shuffle=True)
    return DataLoader(tr,**kwargs), DataLoader(va,**kwargs), DataLoader(te,**kwargs)
