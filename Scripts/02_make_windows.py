#!/usr/bin/env python
"""
02_make_windows.py
==================
Создаёт dataset.jsonl для задачи «посимвольная сегментация».

• Positive окна — центры MACS3-summits (по одному на пик)
• Negative окна — случайные позиции ≥BUFF bp от любого narrowPeak
• Формат записи:
    {chrom, start, window, cell, dna_fa, atac_bw, fe_bw}

Баланс 1:1 (NEG_RATIO можно поменять).
"""

import json, random
from pathlib import Path
from collections import defaultdict

import pandas as pd
import pyBigWig
from intervaltree import Interval, IntervalTree
from tqdm import tqdm

# ---------- параметры ----------------------------------------------------- #
ROOT    = Path(__file__).resolve().parents[1]
WIN     = 1000                # длина окна
BUFF    = 5_000               # зона отчуждения от пика для негатива
NEG_RATIO = 1                 # сколько отрицательных на 1 положительный
PAIRS = pd.read_csv(ROOT / "sample_pairs.tsv", sep="\t")

OUT = ROOT / "data" / "dataset.jsonl"
OUT.parent.mkdir(exist_ok=True)

# ------------------------------------------------------------------------- #
def load_peaks(chip_id):
    """читаем narrowPeak, строим IntervalTree расширенных диапазонов"""
    npk = ROOT / f"peaks/{chip_id}/{chip_id}_peaks.narrowPeak"
    tree = defaultdict(IntervalTree)
    with open(npk) as f:
        for line in f:
            chrom, start, end, *_ = line.split()[:3]
            tree[chrom].add(Interval(int(start)-BUFF, int(end)+BUFF))
    return tree

def chrom_sizes(bw_path):
    bw = pyBigWig.open(str(bw_path))
    sizes = bw.chroms(); bw.close()
    return sizes

# ---------- 1-й проход: считаем total ----------------------------------- #
total_pos = 0
for _, row in PAIRS.iterrows():
    summits = ROOT / f"peaks/{row.chip_id}/{row.chip_id}_summits.bed"
    total_pos += sum(1 for _ in open(summits))
total = total_pos * (1 + NEG_RATIO)
print(f"Окон будет записано: {total}  (pos={total_pos}, neg={total_pos*NEG_RATIO})")

# ---------- 2-й проход: записываем jsonl с прогресс-баром --------------- #
OUT.write_text("")
with OUT.open("a") as fout, tqdm(total=total, unit="windows") as pbar:
    for _, row in PAIRS.iterrows():
        atac_bw = ROOT / f"bw/{row.atac_id}_ATAC.bw"
        fe_bw   = ROOT / f"bw/{row.chip_id}_FE.bw"
        fasta   = ROOT / "hg38.fa"

        # --- Positive ---
        summits = ROOT / f"peaks/{row.chip_id}/{row.chip_id}_summits.bed"
        pos_list = []
        with open(summits) as f:
            for line in f:
                chrom, pos_str, _ = line.split()[:3]
                summit = int(pos_str)
                start  = summit - WIN//2
                pos_list.append((chrom, start))

        # --- Negative ---
        tree = load_peaks(row.chip_id)
        chrom_len = chrom_sizes(atac_bw)
        neg_list = []
        rng = random.Random(42 + int(row.cell_id))
        while len(neg_list) < len(pos_list) * NEG_RATIO:
            chrom = rng.choice(list(chrom_len))
            max_start = chrom_len[chrom] - WIN
            if max_start <= 0: continue
            s = rng.randint(0, max_start)
            if not tree[chrom].overlaps(s, s+WIN):
                neg_list.append((chrom, s))

        # --- запись ---
        for chrom, start in pos_list + neg_list:
            rec = dict(
                cell   = int(row.cell_id),
                chrom  = chrom,
                start  = start,
                window = WIN,
                dna_fa = str(fasta),
                atac_bw= str(atac_bw),
                fe_bw  = str(fe_bw),
                label  = 1 if (chrom,start) in pos_list else 0   # meta-флаг
            )
            fout.write(json.dumps(rec) + "\n")
            pbar.update(1)

print(f"✓ dataset.jsonl создан: {OUT}")
