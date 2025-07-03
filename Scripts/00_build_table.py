#!/usr/bin/env python
"""
Сводим MYC_info.xlsx  → sample_pairs.tsv
Оставляем ровно ОДНУ пару (ChIP + ATAC) на каждый cell_id.
Пара считается валидной, только если реально существует BAM-файл.
Формат выхода: cell_id,chip_id,chip_bam,atac_id,atac_bam
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]          # /mnt/d/ChipSeqAtacBAMS
SRC  = ROOT / "source_bam"                         # <- новая папка с BAM
XLSX = ROOT / "MYC_info.xlsx"
OUT  = ROOT / "sample_pairs.tsv"

chip   = pd.read_excel(XLSX, sheet_name="chipseq_info_MYC")
atac   = pd.read_excel(XLSX, sheet_name="atac_info_MYC")
pairs  = []

def first_valid(df, kind):
    """вернуть первую строку, для которой существует BAM"""
    for _, r in df.iterrows():
        bam = SRC / f"{r.align_id}.bam"
        if bam.exists():
            print(f"[✓ {kind}] {r.id:<11}  {bam.name}")
            return r, bam
        print(f"[✗ {kind}] {r.id:<11}  {bam.name} — нет файла")
    return None, None

for cid in sorted(set(chip.cell_id) | set(atac.cell_id)):
    c_rows = chip.query("cell_id == @cid")
    a_rows = atac.query("cell_id == @cid")
    if c_rows.empty or a_rows.empty:
        print(f"[SKIP] cell_id {cid}: нет данных одновременно ChIP+ATAC в xlsx")
        continue

    c_row, c_bam = first_valid(c_rows, "ChIP")
    a_row, a_bam = first_valid(a_rows, "ATAC")
    if c_bam and a_bam:
        pairs.append(dict(cell_id=cid,
                          chip_id=c_row.id, chip_bam=str(c_bam),
                          atac_id=a_row.id, atac_bam=str(a_bam)))
    else:
        print(f"[SKIP] cell_id {cid}: нет валидной пары файлов\n")

if not pairs:
    raise SystemExit("‼ Ни одной валидной пары не найдено")

pd.DataFrame(pairs).to_csv(OUT, sep="\t", index=False)
print(f"\n✓ sample_pairs.tsv сохранён ({len(pairs)} строк)")
