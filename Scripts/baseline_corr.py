#!/usr/bin/env python
"""
baseline_corr.py
================
«Бейзлайновая корреляция» между входом (ATAC-профиль) и
выходом (маска FE>thr) без всякой нейросети.

Запуск:
    python Scripts/baseline_corr.py --jsonl data/dataset.jsonl \
                            --fe_thresh 1.0 \
                            --out baseline_corr_per_window.csv
"""

import argparse, json, csv, math
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import pyBigWig
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


import warnings, scipy.stats
warnings.filterwarnings("ignore", category=scipy.stats.ConstantInputWarning)

# ------------------------------------------------------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--jsonl", required=True)
parser.add_argument("--fe_thresh", type=float, default=1.0)
parser.add_argument("--out", default="baseline_corr_per_window.csv")
args = parser.parse_args()

# simple caches
_bw_cache = {}
def open_bw(path):
    if path not in _bw_cache:
        _bw_cache[path] = pyBigWig.open(path)
    return _bw_cache[path]

# ------------------------------------------------------------------------- #
rows = []
with open(args.jsonl) as f, tqdm(unit="windows") as pbar:
    for line in f:
        rec = json.loads(line)
        bw_atac = open_bw(rec["atac_bw"])
        bw_fe   = open_bw(rec["fe_bw"])
        L       = int(rec["window"])
        s       = int(rec["start"])
        e       = s + L
        chrom   = rec["chrom"]

        # извлекаем векторы
        atac = np.nan_to_num(bw_atac.values(chrom, s, e), nan=0.0)
        fe   = np.nan_to_num(bw_fe.values(chrom, s, e), nan=0.0)
        mask = (fe > args.fe_thresh).astype(float)

        # Pearson / Spearman; обрабатываем случай const vector
        try:
            pr = pearsonr(atac, mask)[0]
        except ValueError:
            pr = math.nan
        try:
            sr = spearmanr(atac, mask).correlation
        except ValueError:
            sr = math.nan
        # AUROC — если обе метки присутствуют
        if mask.sum() in (0, L):
            au = math.nan
        else:
            au = roc_auc_score(mask, atac)

        rows.append([rec["cell"], chrom, s, pr, sr, au])
        pbar.update(1)

# ------------------------------------------------------------------------- #
# агрегируем
pearsons  = [r[3] for r in rows if not math.isnan(r[3])]
spearmans = [r[4] for r in rows if not math.isnan(r[4])]
aucs      = [r[5] for r in rows if not math.isnan(r[5])]

def agg(v): return f"{mean(v):.4f} ± {pstdev(v):.4f}" if v else "n/a"

print("\n=== Baseline correlation (ATAC vs FE) ===")
print(f"Pearson r  : {agg(pearsons)}")
print(f"Spearman ρ : {agg(spearmans)}")
print(f"AUROC-atac : {agg(aucs)}")

