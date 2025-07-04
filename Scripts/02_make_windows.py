#!/usr/bin/env python
"""
Создаёт dataset.jsonl (positive + negative окна).
Окно добавляется, только если полностью влезает
и в ATAC-BW, и в FE-BW.
"""

import json, random
from pathlib import Path
from collections import defaultdict

import pandas as pd
import pyBigWig
from intervaltree import Interval, IntervalTree
from tqdm import tqdm

ROOT   = Path(__file__).resolve().parents[1]
WIN    = 1000
BUFF   = 5_000
NEG_RATIO = 1
PAIRS  = pd.read_csv(ROOT / "sample_pairs.tsv", sep="\t")
OUT    = ROOT / "data" / "dataset.jsonl"
OUT.parent.mkdir(exist_ok=True)

# ---------- helpers ------------------------------------------------------- #
def chrom_sizes(bw_path):
    bw = pyBigWig.open(str(bw_path)); sizes = bw.chroms(); bw.close()
    return sizes

def load_peak_tree(chip_id):
    npk = ROOT / f"peaks/{chip_id}/{chip_id}_peaks.narrowPeak"
    tree = defaultdict(IntervalTree)
    with open(npk) as f:
        for line in f:
            chrom,s,e,*_ = line.split()[:3]
            tree[chrom].add(Interval(int(s)-BUFF, int(e)+BUFF))
    return tree

def summit_list(chip_id):
    summ = ROOT / f"peaks/{chip_id}/{chip_id}_summits.bed"
    with open(summ) as f:
        for line in f:
            chrom,pos,*_ = line.split()[:2]
            yield chrom, int(pos)

# ---------- count total --------------------------------------------------- #
total_pos = sum(1 for _,r in PAIRS.iterrows()
                  for _ in summit_list(r.chip_id))
TOTAL = total_pos * (1+NEG_RATIO)
print(f"Всего окон: {TOTAL} (pos={total_pos}, neg={total_pos*NEG_RATIO})")

# ---------- write jsonl --------------------------------------------------- #
OUT.write_text("")
with OUT.open("a") as fout, tqdm(total=TOTAL, unit="windows") as pbar:
    for _, row in PAIRS.iterrows():
        atac_bw = ROOT / f"bw/{row.atac_id}_ATAC.bw"
        fe_bw   = ROOT / f"bw/{row.chip_id}_FE.bw"
        fasta   = ROOT / "hg38.fa"

        sizes_atac = chrom_sizes(atac_bw)
        sizes_fe   = chrom_sizes(fe_bw)
        sizes = {c:min(sizes_atac.get(c,0), sizes_fe.get(c,0))
                 for c in sizes_atac.keys() & sizes_fe.keys()}

        # positive
        pos = []
        for chrom,posi in summit_list(row.chip_id):
            if chrom not in sizes: continue
            start = posi - WIN//2
            if 0 <= start <= sizes[chrom]-WIN:
                pos.append((chrom,start))

        # negative
        tree = load_peak_tree(row.chip_id)
        neg = []
        rng = random.Random(42+int(row.cell_id))
        while len(neg) < len(pos)*NEG_RATIO:
            chrom = rng.choice(list(sizes))
            max_s = sizes[chrom]-WIN
            if max_s<=0: continue
            s = rng.randint(0,max_s)
            if not tree[chrom].overlaps(s,s+WIN):
                neg.append((chrom,s))

        # write
        for chrom,start in pos+neg:
            rec = dict(
                cell=int(row.cell_id), chrom=chrom, start=start,
                window=WIN,
                dna_fa=str(fasta),
                atac_bw=str(atac_bw),
                fe_bw=str(fe_bw),
                label=1 if (chrom,start) in pos else 0
            )
            fout.write(json.dumps(rec)+"\n")
            pbar.update(1)

print(f"✓ dataset.jsonl готов: {OUT}")
