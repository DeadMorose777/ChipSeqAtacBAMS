#!/usr/bin/env python3
"""
Собирает sample_table.tsv   (ChIP + ATAC)
* первичный ключ – cell_id  (cell_type)  
* если в каком-то листе >1 записи на cell_id, берётся первая
* для каждой ChIP-строки подставляется atac_sample_id – первый ATAC
  с тем же cell_id; если его нет – пишем NA
"""
import pandas as pd
import pathlib, sys, os

ROOT   = pathlib.Path(__file__).resolve().parents[1]          # /mnt/d/ChipSeqAtacBAMS
XLSX   = ROOT / "MYC_info.xlsx"
OUT    = ROOT / "sample_table.tsv"

chip = pd.read_excel(XLSX, sheet_name="chipseq_info_MYC")
atac = pd.read_excel(XLSX, sheet_name="atac_info_MYC")

# оставляем по одной записи на каждый cell_id (первая встреченная)
chip = chip.drop_duplicates(subset="cell_id", keep="first").copy()
atac = atac.drop_duplicates(subset="cell_id", keep="first").copy()

# словарь: cell_id → ATAC-sample_id
cell2atac = dict(zip(atac["cell_id"], atac["id"]))

records = []

for _, row in chip.iterrows():
    cid  = row["cell_id"]
    atac_id = cell2atac.get(cid, pd.NA)

    records.append({
        "sample_id"      : row["id"],
        "assay"          : "ChIP",
        "species"        : row["species"],
        "treatment"      : row["treatment"],
        "cell_id"        : cid,
        "bam_path"       : str(ROOT / f"{row['align_id']}.bam"),
        "control_bam"    : str(ROOT / f"{row['control_id']}.bam") if pd.notna(row["control_id"]) else pd.NA,
        "atac_sample_id" : atac_id
    })

for _, row in atac.iterrows():
    records.append({
        "sample_id"      : row["id"],
        "assay"          : "ATAC",
        "species"        : row["species"],
        "treatment"      : row["treatment"],
        "cell_id"        : row["cell_id"],
        "bam_path"       : str(ROOT / f"{row['align_id']}.bam"),
        "control_bam"    : pd.NA,
        "atac_sample_id" : pd.NA
    })

tbl = pd.DataFrame.from_records(records)
tbl.to_csv(OUT, sep="\t", index=False)

print(f"✓ sample_table.tsv переписан: {OUT}  (строк: {len(tbl)})")
