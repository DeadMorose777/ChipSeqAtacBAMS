#!/usr/bin/env python3
"""
Создаёт dataset.jsonl со скользящими окнами.
Использует sample_table.tsv, колонку atac_sample_id.
Если bigWig ATAC для данного ChIP не найден, окно пропускается,
но скрипт продолжает работу.
"""
import json, random, pathlib, os, multiprocessing as mp, pyBigWig, pyfaidx, sys, warnings
import pandas as pd

ROOT   = pathlib.Path(__file__).resolve().parents[1]
GENOME = os.environ.get("GENOME", str(ROOT / "hg38.fa"))
FA     = pyfaidx.Fasta(GENOME)
TAB    = pd.read_csv(ROOT / "sample_table.tsv", sep="\t")

WIN = 512           # длина ДНК-окна
STEP= 256           # шаг

# словарь ATAC-bw: sample_id → путь
bw_atac = { r["sample_id"]: str(ROOT/f"bw/{r['sample_id']}_ATAC.bw")
            for _,r in TAB.query("assay=='ATAC'").iterrows()
            if os.path.exists(ROOT/f"bw/{r['sample_id']}_ATAC.bw") }

peaks_dir = ROOT / "peaks"
out_path  = ROOT / "dataset.jsonl"

def one_sample(row):
    cid   = row["cell_id"]
    atac  = row["atac_sample_id"]
    if pd.isna(atac) or atac not in bw_atac:
        warnings.warn(f"⚠️  Нет подходящего ATAC для {row['sample_id']} (cell {cid})")
        return []

    bw     = pyBigWig.open(bw_atac[atac])
    peaks  = pathlib.Path(peaks_dir/row["sample_id"]/f"{row['sample_id']}_summits.bed")
    if not peaks.exists():
        warnings.warn(f"⚠️  Нет summits.bed для {row['sample_id']}")
        return []

    windows = []
    with peaks.open() as bed:
        for ln in bed:
            chrom, pos = ln.split()[:2]
            center = int(pos)
            start  = max(0, center - WIN//2)
            seq    = str(FA[chrom][start:start+WIN]).upper()
            atac_val = bw.stats(chrom, start, start+WIN, type="mean")[0] or 0
            windows.append({
                "id"    : f"{row['sample_id']}:{chrom}:{start}-{start+WIN}",
                "seq"   : seq,
                "atac"  : atac_val,
                "label" : 1                       # положительное окно
            })
    bw.close()
    return windows

# собираем для всех ChIP
chip_rows = TAB.query("assay=='ChIP'")
with mp.Pool() as pool, open(out_path,"w") as out:
    for wins in pool.imap_unordered(one_sample, chip_rows.to_dict("records")):
        for w in wins:
            out.write(json.dumps(w)+"\n")

print(f"✓ dataset.jsonl готов: {out_path}")
