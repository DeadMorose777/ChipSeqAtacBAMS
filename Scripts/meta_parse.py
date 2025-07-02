#!/usr/bin/env python3
"""
Из MYC_info.xlsx собирает таблицу sample_table.tsv
колонки важны: sample_id, assay (ChIP/ATAC), bam_path, control_bam
"""
import pandas as pd, pathlib

ROOT      = pathlib.Path(__file__).resolve().parents[1]      # /mnt/d/ChipSeqAtacBAMS
XLSX      = ROOT / "MYC_info.xlsx"
OUT_TSV   = ROOT / "sample_table.tsv"

xl = pd.read_excel(XLSX, sheet_name=None)
df = pd.concat(xl.values(), ignore_index=True)

# выбираем только валидные (столбец 'use' = 1)
df = df[df.get("use", 1) == 1].copy()

# абсолютные пути к BAM
df["bam_path"]    = df["bam_file"].apply(lambda f: str(ROOT / f))
df["control_bam"] = df["control_bam"].apply(lambda f: str(ROOT / f) if isinstance(f, str) else "")

df[["sample_id","assay","bam_path","control_bam"]].to_csv(OUT_TSV, sep="\t", index=False)
print("✓ sample_table.tsv создан:", OUT_TSV)
