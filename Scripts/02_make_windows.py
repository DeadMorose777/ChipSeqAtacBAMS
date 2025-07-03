#!/usr/bin/env python
"""
02_make_windows.py
Готовит dataset.jsonl из пар sample_pairs.tsv.
"""

import json, random, warnings
from pathlib import Path, PurePath
import pandas as pd
import pyBigWig, pyfaidx

ROOT = Path(__file__).resolve().parents[1]
WIN  = 200                         # размер окна

DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

pairs = pd.read_csv(ROOT / "sample_pairs.tsv", sep="\t")
fa    = pyfaidx.Fasta(str(ROOT / "hg38.fa"))

# ------------ вспом-функции -------------------------------------------------
def chr_len(chrom: str) -> int:
    return len(fa[chrom])

def ok_interval(chrom: str, start: int, bw: pyBigWig.pyBigWig) -> bool:
    """Хромосома есть в BigWig и окно [start, start+WIN) внутри неё."""
    if chrom not in bw.chroms():
        return False
    return 0 <= start and start + WIN < bw.chroms()[chrom]

# ------------ работа --------------------------------------------------------
dst = DATA_DIR / "dataset.jsonl"
dst.write_text("")

for _, row in pairs.iterrows():
    cell    = int(row.cell_id)
    chip_bw = pyBigWig.open(str(ROOT / f"bw/{row.chip_id}_FE.bw"))
    atac_bw = pyBigWig.open(str(ROOT / f"bw/{row.atac_id}_ATAC.bw"))

    common_chroms = set(chip_bw.chroms()) & set(atac_bw.chroms())
    if not common_chroms:
        warnings.warn(f"🟥  cell {cell}: нет общих хромосом между BW → пропуск")
        continue

    summits = ROOT / f"peaks/{row.chip_id}/{row.chip_id}_summits.bed"
    if not summits.exists():
        warnings.warn(f"🟥  нет summits для {row.chip_id}")
        continue

    peak_coords = []
    with summits.open() as fh:
        for ln in fh:
            if not ln.strip():
                continue
            chrom, summit, *_ = ln.split()
            summit = int(summit)
            start  = summit - WIN // 2
            if chrom in common_chroms and ok_interval(chrom, start, chip_bw):
                peak_coords.append((chrom, start))

    if not peak_coords:
        warnings.warn(f"⚠️  все summits вышли за пределы BW для {row.chip_id}")
        continue

    with dst.open("a") as fout:
        # ------- positive windows -------
        for chrom, start in peak_coords:
            try:
                chip = chip_bw.stats(chrom, start, start + WIN, type="mean")[0] or 0
                atac = atac_bw.stats(chrom, start, start + WIN, type="mean")[0] or 0
            except RuntimeError as e:
                warnings.warn(f"Skip bad summit {chrom}:{start}-{start+WIN} ({e})")
                continue
            fout.write(json.dumps(dict(cell=cell, chrom=chrom, start=start,
                                       chip=round(chip,3), atac=round(atac,3),
                                       label=1)) + "\n")
        n_pos = len(peak_coords)

        # ------- negative windows -------
        rng_chroms = list(common_chroms)
        for _ in range(n_pos):
            while True:
                chrom = random.choice(rng_chroms)
                max_start = chip_bw.chroms()[chrom] - WIN - 1
                start = random.randint(0, max_start)
                if ok_interval(chrom, start, chip_bw):
                    break
            chip = chip_bw.stats(chrom, start, start + WIN, type="mean")[0] or 0
            atac = atac_bw.stats(chrom, start, start + WIN, type="mean")[0] or 0
            fout.write(json.dumps(dict(cell=cell, chrom=chrom, start=start,
                                       chip=round(chip,3), atac=round(atac,3),
                                       label=0)) + "\n")

    print(f"✓ cell {cell}: {n_pos} peaks + {n_pos} bg")

chip_bw.close(); atac_bw.close()

size_mb = dst.stat().st_size / 1e6
print(f"\n⇢ {dst.relative_to(ROOT)} готов — {size_mb:.1f} MB")