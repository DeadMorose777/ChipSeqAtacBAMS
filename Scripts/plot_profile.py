#!/usr/bin/env python
"""
plot_profile.py  --chip EXP057893 --chr chr8 --start 127735000 --end 127740000
"""
import argparse, pyBigWig, matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--chip",  required=True)  # chip_id без суффиксов
ap.add_argument("--chr",   required=True)
ap.add_argument("--start", type=int, required=True)
ap.add_argument("--end",   type=int, required=True)
args = ap.parse_args()

root = Path(__file__).resolve().parents[1] / "bw"
bw_real = pyBigWig.open(str(root / f"{args.chip}_FE.bw"))
bw_pred = pyBigWig.open(str(root / f"{args.chip}_pred.bw"))

x = list(range(args.start, args.end))
y_real = bw_real.values(args.chr, args.start, args.end)
y_pred = bw_pred.values(args.chr, args.start, args.end)

plt.figure(figsize=(10,3))
plt.plot(x, y_real,  label="MACS3 FE")
plt.plot(x, y_pred,  label="Predicted", alpha=0.7)
plt.title(f"{args.chip}  {args.chr}:{args.start}-{args.end}")
plt.xlabel("genomic position"); plt.ylabel("signal")
plt.legend(); plt.tight_layout(); plt.show()
