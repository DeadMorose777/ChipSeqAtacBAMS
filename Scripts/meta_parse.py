#!/usr/bin/env python3
"""
Читает MYC_info.xlsx  →  sample_table.tsv
Оставляет по одной записи на cell_id.
Содержит только:
    cell_id, chip_bw, atac_bw
"""
import pandas as pd, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
xls  = ROOT / "MYC_info.xlsx"
out  = ROOT / "sample_table.tsv"

dfc = pd.read_excel(xls, sheet_name="chipseq_info_MYC")
dfa = pd.read_excel(xls, sheet_name="atac_info_MYC")

# оставляем по одной строке на cell_id
dfc = dfc.drop_duplicates("cell_id", keep="first")
dfa = dfa.drop_duplicates("cell_id", keep="first")

tbl = dfc.merge(dfa[["cell_id","id"]], on="cell_id", how="left",
                suffixes=("_chip","_atac"))

records = []
for r in tbl.itertuples():
    sid      = r.id_chip      # EXP....
    aid      = r.id_atac      # AEXP....
    chip_bw  = ROOT / f"bw/{sid}_FE.bw"
    atac_bw  = ROOT / f"bw/{aid}_ATAC.bw" if pd.notna(aid) else None
    records.append((r.cell_id, chip_bw, atac_bw))

pd.DataFrame(records, columns=["cell_id","chip_bw","atac_bw"])\
  .to_csv(out, sep="\t", index=False)
print(f"✓ sample_table.tsv переписан ({len(records)} строк)")
