#!/usr/bin/env python
"""
02_make_windows.py
------------------
Готовит dataset.jsonl для задачи сегментации пиков.

• 1-й проход  — считаем, сколько окон получится (total)
• 2-й проход  — пишем jsonl, продвигаем tqdm(total)

Окно фиксированной длины WIN вокруг центра каждого MACS-пика.
"""

import json
from pathlib import Path

import pandas as pd
import pyBigWig
from tqdm import tqdm


# ---------- константы ----------------------------------------------------- #
ROOT    = Path(__file__).resolve().parents[1]
WIN     = 1000             # длина окна, bp
THRESH  = 1.0              # FE > THRESH ⇒ пиковая позиция
PAIRS_TSV = ROOT / "sample_pairs.tsv"

OUT_PATH = ROOT / "data" / "dataset.jsonl"
OUT_PATH.parent.mkdir(exist_ok=True)


# ---------- utils --------------------------------------------------------- #
def intervals_from_bw(bw: pyBigWig.pyBigWig, chrom: str):
    """список (start,end) где FE>THRESH"""
    vals = bw.values(chrom, 0, bw.chroms()[chrom], numpy=False)
    starts, ends, in_peak = [], [], False
    for i, v in enumerate(vals):
        v = v or 0.0
        if v > THRESH and not in_peak:
            in_peak, s = True, i
        elif v <= THRESH and in_peak:
            in_peak = False
            starts.append(s); ends.append(i)
    if in_peak:
        starts.append(s); ends.append(bw.chroms()[chrom]-1)
    return zip(starts, ends)


def count_windows(pairs_df):
    """быстрый подсчёт общего числа окон"""
    total = 0
    for _, row in pairs_df.iterrows():
        fe_bw = pyBigWig.open(str(ROOT / f"bw/{row.chip_id}_FE.bw"))
        atac  = pyBigWig.open(str(ROOT / f"bw/{row.atac_id}_ATAC.bw"))
        for chrom in (set(fe_bw.chroms()) & set(atac.chroms())):
            for s, e in intervals_from_bw(fe_bw, chrom):
                center = (s + e)//2 - WIN//2
                if 0 <= center < fe_bw.chroms()[chrom] - WIN:
                    total += 1
        fe_bw.close(); atac.close()
    return total


# ---------- main ---------------------------------------------------------- #
pairs = pd.read_csv(PAIRS_TSV, sep="\t")
TOTAL = count_windows(pairs)
print(f"Окон будет записано: {TOTAL}")

OUT_PATH.write_text("")
with OUT_PATH.open("a") as fout, tqdm(total=TOTAL, unit="windows") as pbar:
    for _, row in pairs.iterrows():
        atac_bw = ROOT / f"bw/{row.atac_id}_ATAC.bw"
        fe_bw   = ROOT / f"bw/{row.chip_id}_FE.bw"
        atac = pyBigWig.open(str(atac_bw))
        fe   = pyBigWig.open(str(fe_bw))

        for chrom in (set(atac.chroms()) & set(fe.chroms())):
            for s, e in intervals_from_bw(fe, chrom):
                center = (s + e)//2 - WIN//2
                if center < 0 or center + WIN >= fe.chroms()[chrom]:
                    continue
                rec = dict(
                    cell    = int(row.cell_id),
                    chrom   = chrom,
                    start   = center,
                    window  = WIN,
                    dna_fa  = str(ROOT / "hg38.fa"),
                    atac_bw = str(atac_bw),
                    fe_bw   = str(fe_bw),
                )
                fout.write(json.dumps(rec) + "\n")
                pbar.update(1)

        atac.close(); fe.close()

print(f"✓ dataset.jsonl готов: {OUT_PATH}")
