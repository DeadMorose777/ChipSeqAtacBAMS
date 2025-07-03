#!/usr/bin/env python3
"""
Берём sample_table.tsv → формируем dataset.jsonl.

 • окно WIN (по умолчанию 200 bp) скользит по узким пикам ChIP-seq
 • для каждого окна пишем JSON-строку
       {"cell_id":…, "seq":…, "atac":…, "chip":…}

   – seq   : 200-bp-последовательность из hg38
   – atac  : mean-coverage по ATAC-BW
   – chip  : mean Fold-Enrichment по ChIP-BW  (целевая переменная)

 Все строки пишутся в gzip-архив `dataset.jsonl`.
 Файл пропускается, если не удалось открыть соответствующий bw
 или окно выходит за границы хромосомы.
"""
import os, json, gzip, warnings, pathlib
import multiprocessing as mp

import pandas as pd, numpy as np, pyBigWig, pyfaidx

WIN   = 200
ROOT  = pathlib.Path(__file__).resolve().parents[1]
GENOME= pathlib.Path(os.getenv("GENOME", ROOT / "hg38.fa"))
TSV   = ROOT / "sample_table.tsv"
OUT   = ROOT / "dataset.jsonl"

# ------------------ входные файлы ------------------
fa  = pyfaidx.Fasta(str(GENOME))

tbl = pd.read_csv(TSV, sep="\t")
# пытаемся открыть BigWig-и; если файла нет – клетка отбрасывается
bw_at, bw_ch = {}, {}
keep_rows = []

for r in tbl.itertuples():
    ok = True
    try:
        bw_at[r.cell_id] = pyBigWig.open(r.atac_bw)
    except Exception as e:
        warnings.warn(f"🟥  нет ATAC-BW {r.atac_bw} ({e})")
        ok = False
    try:
        bw_ch[r.cell_id] = pyBigWig.open(r.chip_bw)
    except Exception as e:
        warnings.warn(f"🟥  нет ChIP-BW {r.chip_bw} ({e})")
        ok = False
    if ok:
        keep_rows.append(r)

if not keep_rows:
    raise SystemExit("❌  Ни одной пары ATAC+ChIP BigWig не открылась")

print(f"✓ к обработке {len(keep_rows)} клеток "
      f"(из {len(tbl)}) – остальные отброшены")

# ----------------- вспом-функции -------------------
def peaks_path(chip_bw_path:str)->pathlib.Path:
    samp = pathlib.Path(chip_bw_path).stem.replace("_FE","")
    return ROOT / f"peaks/{samp}/{samp}_summits.bed"

def one_sample(r):
    cid   = r.cell_id
    peaks = peaks_path(r.chip_bw)
    res   = []

    if not peaks.exists():
        warnings.warn(f"⚠️  нет пиков {peaks}")
        return res

    atac = bw_at[cid]
    chip = bw_ch[cid]

    with peaks.open() as fh:
        for ln in fh:
            chrom, start, *_ = ln.split()[:3]
            start  = int(start)
            center = start + WIN // 2
            left   = max(0, center - WIN // 2)
            right  = left + WIN

            # обрезаем окна, выходящие за конец хромосомы
            chr_len = len(fa[chrom])
            if right > chr_len:
                continue

            seq = fa[chrom][left:right].seq.upper()
            if len(seq) != WIN or "N" in seq:
                continue

            atac_val = atac.stats(chrom, left, right, type="mean")[0] or 0
            chip_val = chip.stats(chrom, left, right, type="mean")[0] or 0

            res.append({
                "cell_id": int(cid),
                "seq"    : seq,
                "atac"   : atac_val,
                "chip"   : chip_val
            })
    return res

# --------------- параллельная выгрузка --------------
with mp.Pool() as pool, gzip.open(OUT, "wt") as gz:
    total = 0
    for recs in pool.imap_unordered(one_sample, keep_rows):
        for r in recs:
            gz.write(json.dumps(r) + "\n")
            total += 1

print(f"✓ dataset.jsonl готов – {total} окон")
