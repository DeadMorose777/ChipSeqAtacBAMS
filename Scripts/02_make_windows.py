#!/usr/bin/env python
"""
Создаём dataset.jsonl из sample_pairs.tsv.
Одно окно (WIN=200 bp) → одна строка json:
{"cell":937,"chrom":"chr1","start":123456,...,"chip":0.37,"atac":3.2}
"""

import json, random, warnings
from pathlib import Path
import pandas as pd, pyBigWig, pyfaidx, numpy as np

ROOT = Path(__file__).resolve().parents[1]
WIN  = 200
pairs = pd.read_csv(ROOT/'sample_pairs.tsv', sep='\t')

fa   = pyfaidx.Fasta(str(ROOT/'hg38.fa'))          # для случайных фоновых окон
lengths = {c.len for c in fa.records.values()}

dst = ROOT/'dataset.jsonl'
dst.write_text("")          # очищаем

for _, row in pairs.iterrows():
    cell = row.cell_id
    chip_bw = ROOT/f"bw/{row.chip_id}_FE.bw"
    atac_bw = ROOT/f"bw/{row.atac_id}_ATAC.bw"

    try:
        bw_c = pyBigWig.open(str(chip_bw));  bw_a = pyBigWig.open(str(atac_bw))
    except RuntimeError as e:
        warnings.warn(f"🟥  cell {cell}: BigWig открыть не удалось → пропуск ({e})")
        continue

    # ------- реальные пики (используем summits.bed) -------
    summits = ROOT/f"peaks/{row.chip_id}/{row.chip_id}_summits.bed"
    if not summits.exists():
        warnings.warn(f"🟥  нет summits для {row.chip_id} → пропуск клетки")
        continue

    for ln in summits.open():
        chrom, summ, *_ = ln.split()
        start = int(summ) - WIN//2
        if start < 0: continue
        chip_val = bw_c.stats(chrom, start, start+WIN, type="mean")[0] or 0
        atac_val = bw_a.stats(chrom, start, start+WIN, type="mean")[0] or 0
        rec = dict(cell=int(cell), chrom=chrom, start=start,
                   chip=round(chip_val,3), atac=round(atac_val,3))
        dst.open("a").write(json.dumps(rec)+"\n")

    # ------- фоновый контроль -------
    for _ in range( len(open(summits).readlines()) ):
        chrom = random.choice(list(fa.keys()))
        max_start = fa[chrom].end - WIN - 1
        start = random.randint(0, max_start)
        chip_val = bw_c.stats(chrom, start, start+WIN, type="mean")[0] or 0
        atac_val = bw_a.stats(chrom, start, start+WIN, type="mean")[0] or 0
        rec = dict(cell=int(cell), chrom=chrom, start=start,
                   chip=round(chip_val,3), atac=round(atac_val,3),
                   background=1)
        dst.open("a").write(json.dumps(rec)+"\n")

    print(f"✓ windows для cell {cell}: {row.chip_id}/{row.atac_id}")

print(f"\n⇢ dataset.jsonl готов: {dst.stat().st_size/1e6:.1f} MB")
