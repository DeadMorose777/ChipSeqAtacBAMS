#!/usr/bin/env python3
"""
Скользящее окно → BigWig с предсказанным ChIP-FE/логитами.
Работает с новой 5-канальной архитектурой: seq-строка + ATAC-вектор(L).
"""
import argparse, pyBigWig, pyfaidx, torch, yaml, numpy as np
from pathlib import Path
from src.registry import get_model_cls

# ------------------------- CLI ------------------------------------------ #
ap = argparse.ArgumentParser()
ap.add_argument("--model_dir", required=True)
ap.add_argument("--fasta",     required=True)
ap.add_argument("--atac",      required=True)
ap.add_argument("--outbw",     required=True)
ap.add_argument("--window", type=int, default=1000)
ap.add_argument("--step",   type=int, default=50)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

# ------------------------- load model ----------------------------------- #
run_cfg   = yaml.safe_load(open(Path(args.model_dir) / "run_config.json"))
model_cfg = yaml.safe_load(open(run_cfg["model_cfg"]))
ModelCls  = get_model_cls(Path(run_cfg["model_cfg"]).stem)
model     = ModelCls(model_cfg).to(args.device)
model.load_state_dict(torch.load(Path(args.model_dir) / "best.pt",
                                 map_location=args.device))
model.eval()

# ------------------------- genome I/O ----------------------------------- #
fa        = pyfaidx.Fasta(args.fasta)
bw_atac   = pyBigWig.open(args.atac)
bw_pred   = pyBigWig.open(args.outbw, "w")
chroms    = {c: len(fa[c]) for c in fa.keys()}
bw_pred.addHeader(list(chroms.items()))

def atac_vector(chrom, s, e):
    vec = bw_atac.values(chrom, s, e)
    return np.nan_to_num(vec, nan=0.0).astype("float32")           # (L,)

@torch.no_grad()
def score(chrom, s, e):
    seq  = fa[chrom][s:e].seq.upper()
    atac = torch.tensor(atac_vector(chrom, s, e))[None, :]         # (1,L)
    rec  = {"seq": seq, "atac": atac}
    batch = model.collate_fn([rec])
    for k,v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(args.device)
    y = model(batch).sigmoid().squeeze().cpu().item()              # prob ∈(0,1)
    return y

# ------------------------- sliding window ------------------------------- #
for chrom, length in chroms.items():
    starts, ends, vals = [], [], []
    for s in range(0, length - args.window, args.step):
        y = score(chrom, s, s + args.window)
        starts.append(s); ends.append(s + args.window); vals.append(y)
    bw_pred.addEntries([chrom]*len(starts), starts, ends=ends, values=vals)

bw_pred.close(); bw_atac.close(); fa.close()
print("✓ BigWig создан:", args.outbw)
