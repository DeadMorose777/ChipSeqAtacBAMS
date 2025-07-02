#!/usr/bin/env python3
"""
Генерирует sample_table.tsv из MYC_info.xlsx

Если указанного BAM-файла нет — строка пропускается,
в консоль выводится предупреждение.
"""
import pandas as pd, pathlib, sys

ROOT   = pathlib.Path(__file__).resolve().parents[1]
XLSX   = ROOT / "MYC_info.xlsx"
OUT    = ROOT / "sample_table.tsv"

try:
    chip = pd.read_excel(XLSX, sheet_name="chipseq_info_MYC")
    atac = pd.read_excel(XLSX, sheet_name="atac_info_MYC")
except ValueError as e:
    sys.exit(f"❌  В {XLSX} не найдены нужные листы: {e}")

def have_bam(row_id: str) -> pathlib.Path | None:
    p = ROOT / f"{row_id}.bam"
    if p.exists():
        return p
    print(f"⚠️  нет BAM: {p.name}", file=sys.stderr)
    return None

rows = []

# --- ChIP ---
for _, r in chip.iterrows():
    bam = have_bam(r["align_id"])
    if not bam:
        continue
    ctrl = have_bam(r["control_id"]) if pd.notna(r["control_id"]) else None
    rows.append(dict(
        sample_id   = r["id"],
        assay       = "ChIP",
        bam_path    = str(bam),
        control_bam = str(ctrl) if ctrl else ""
    ))

# --- ATAC ---
for _, r in atac.iterrows():
    bam = have_bam(r["align_id"])
    if not bam:
        continue
    rows.append(dict(
        sample_id   = r["id"],
        assay       = "ATAC",
        bam_path    = str(bam),
        control_bam = ""
    ))

df = pd.DataFrame(rows)
df.to_csv(OUT, sep="\t", index=False)
print(f"✓ sample_table.tsv создан: {OUT}  (строк: {len(df)})")
