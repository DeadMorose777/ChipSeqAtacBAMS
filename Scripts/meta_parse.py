#!/usr/bin/env python3
"""
DEBUG-версия генератора sample_table.tsv
ничего не меняет на диске, просто пошагово объясняет,
почему cell-линию берём / отбрасываем
"""
import pathlib, pandas as pd, os, shutil, textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]
xls  = ROOT / "MYC_info.xlsx"
bwdir = ROOT / "bw"

dfc = pd.read_excel(xls, sheet_name="chipseq_info_MYC")
dfa = pd.read_excel(xls, sheet_name="atac_info_MYC")

def have(p: pathlib.Path) -> bool:
    return p.exists() and p.stat().st_size > 0

kept, skipped = [], []

for cid, chip_grp in dfc.groupby("cell_id"):
    atac_grp = dfa[dfa.cell_id == cid]

    # ---------- ищем ChIP-BW ----------
    chip_bw = None
    for _, rec in chip_grp.iterrows():
        cand = bwdir / f"{rec.id}_FE.bw"
        if have(cand):
            chip_bw = cand; why_chip = f"нашёл {cand.name}"
            break
    else:
        why_chip = "ни одного *_FE.bw нет"
    
    # ---------- ищем ATAC-BW ----------
    atac_bw = None
    for _, rec in atac_grp.iterrows():
        cand = bwdir / f"{rec.id}_ATAC.bw"
        if have(cand):
            atac_bw = cand; why_atac = f"нашёл {cand.name}"
            break
    else:
        why_atac = "ни одного *_ATAC.bw нет"

    if chip_bw and atac_bw:
        kept.append((cid, chip_bw.name, atac_bw.name))
        status = "✓ KEEP"
    else:
        skipped.append(cid)
        status = "✗ SKIP"

    print(f"[{status}] cell_id={cid:<6}  "
          f"ChIP: {why_chip:<25} | ATAC: {why_atac}")

print("\nИТОГО:")
print("  оставлено :", len(kept))
print("  пропущено :", len(skipped))
print("  файлов BW :", len(list(bwdir.glob('*.bw'))))
print("\nСписок сохранённых пар:")
print(textwrap.indent("\n".join(f"{cid}\t{c}\t{a}" for cid,c,a in kept),"  "))
