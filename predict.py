#!/usr/bin/env python3
"""
Скользящее окно → BigWig с предсказанным ChIP-FE.
"""
import argparse, pyBigWig, pyfaidx, torch, yaml
from pathlib import Path
from src.registry import get_model_cls
from src.layers.tokenizers import DNATokenizer, OneHotEncoder

ap = argparse.ArgumentParser()
ap.add_argument("--model_dir", required=True, help="папка с best.pt и run_config.json")
ap.add_argument("--fasta", required=True)
ap.add_argument("--atac", required=True)
ap.add_argument("--outbw", required=True)
ap.add_argument("--window", type=int, default=500)
ap.add_argument("--step", type=int, default=50)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

run_cfg = yaml.safe_load(open(Path(args.model_dir) / "run_config.json"))
model_cfg = yaml.safe_load(open(run_cfg["model_cfg"]))
ModelCls = get_model_cls(Path(run_cfg["model_cfg"]).stem)
model = ModelCls(model_cfg).to(args.device)
model.load_state_dict(torch.load(Path(args.model_dir) / "best.pt", map_location=args.device))
model.eval()

fa = pyfaidx.Fasta(args.fasta)
bw_atac = pyBigWig.open(args.atac)
bw_out = pyBigWig.open(args.outbw, "w")
chroms = {c: len(fa[c]) for c in fa.keys()}
bw_out.addHeader(list(chroms.items()))

def mean_atac(chr, s, e):
    return bw_atac.stats(chr, s, e, type="mean")[0] or 0.0

@torch.no_grad()
def score(chr, s, e):
    seq = fa[chr][s:e].seq.upper()
    rec = {"seq": seq, "atac": torch.tensor(mean_atac(chr, s, e))}
    batch = model.collate_fn([rec])
    for k in batch:
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(args.device)
    y = model(batch).item()
    return y

for chr, length in chroms.items():
    vals = []
    for start in range(0, length - args.window, args.step):
        end = start + args.window
        y = score(chr, start, end)
        vals.append((chr, start, end, y))
    bw_out.addEntries([v[0] for v in vals],
                      [v[1] for v in vals],
                      ends=[v[2] for v in vals],
                      values=[v[3] for v in vals])
bw_out.close()
print("✓ BigWig создан:", args.outbw)
