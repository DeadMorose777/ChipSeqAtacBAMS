#!/usr/bin/env python3
"""
predict.py  (v2)
================
• читает список окон из --jsonl;
• для каждого окна и его ±extra соседей (см. --extra) считает вектор
  вероятностей с моделью CNN/Transformer/АТАС-only/ДНК-only;
• усредняет перекрытия и пишет BigWig.

пример:
python predict.py \
    --model_dir runs/transf_run1 \
    --fasta hg38.fa \
    --atac  bw/AEXP001289_ATAC.bw \
    --jsonl data/dataset.jsonl \
    --outbw bw/EXP057893_pred.bw \
    --extra 2 \
    --device cuda
"""
import argparse, json, yaml
from pathlib import Path
import numpy as np, torch, pyBigWig, pyfaidx
from tqdm import tqdm
from src.registry import get_model_cls

# ---------------- CLI ----------------
ap = argparse.ArgumentParser()
ap.add_argument("--model_dir", required=True)
ap.add_argument("--fasta",     required=True)
ap.add_argument("--atac",      required=True)
ap.add_argument("--jsonl",     required=True)
ap.add_argument("--outbw",     required=True)
ap.add_argument("--extra",type=int,default=0,
                help="сколько соседних окон +- брать дополнительно")
ap.add_argument("--device",    default="cuda")
args = ap.parse_args()

# ---------------- MODEL --------------
run_cfg   = yaml.safe_load(open(Path(args.model_dir)/"run_config.json"))
model_cfg = yaml.safe_load(open(run_cfg["model_cfg"]))
ModelCls  = get_model_cls(Path(run_cfg["model_cfg"]).stem)
model     = ModelCls(model_cfg).to(args.device)
model.load_state_dict(torch.load(Path(args.model_dir)/"best.pt",
                                 map_location=args.device))
model.eval()

# ---------------- GENOME I/O ---------
fa   = pyfaidx.Fasta(args.fasta)
bw_a = pyBigWig.open(args.atac)
chrom_len = {c: len(fa[c]) for c in fa.keys()}

# аккумуляторы
sum_dict = {c: np.zeros(l, np.float32)  for c,l in chrom_len.items()}
cnt_dict = {c: np.zeros(l, np.uint16)   for c,l in chrom_len.items()}

def atac_vec(ch,s,e):
    v = bw_a.values(ch,s,e)
    return np.nan_to_num(v,nan=0.0).astype("float32")

@torch.no_grad()
def window_logits(seq, atac_np):
    atac = torch.tensor(atac_np)[None,:]
    rec  = {"seq":seq, "atac":atac, "label":torch.zeros(len(seq))}
    batch= model.collate_fn([rec])
    for k,v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k]=v.to(args.device)
    return model(batch).sigmoid().squeeze(0).cpu().numpy()

# ---------------- ACCUMULATE ---------
# считаем реальное число окон с учётом extra
base_windows = [json.loads(l) for l in open(args.jsonl)]
total_win = len(base_windows)*(2*args.extra+1)
pbar = tqdm(total=total_win, unit="win")

for rec in base_windows:
    chrom = rec["chrom"]
    base_start = int(rec["start"])
    L = int(rec.get("window",1000))
    for shift in range(-args.extra, args.extra+1):
        start = base_start + shift*L
        end   = start + L
        if start < 0 or chrom not in chrom_len or end > chrom_len[chrom]:
            pbar.update(1); continue
        seq  = fa[chrom][start:end].seq.upper()
        atac = atac_vec(chrom,start,end)
        log  = window_logits(seq, atac)

        sum_dict[chrom][start:end] += log
        cnt_dict[chrom][start:end] += 1
        pbar.update(1)
pbar.close()

# ---------------- WRITE BigWig -------
bw_out = pyBigWig.open(args.outbw,"w")
bw_out.addHeader(list(chrom_len.items()))

bases_total = sum((cnt>0).sum() for cnt in cnt_dict.values())
with tqdm(total=bases_total, unit="bp") as bar:
    for chrom,length in chrom_len.items():
        cnt  = cnt_dict[chrom]
        mask = cnt>0
        if not mask.any():
            continue
        prob = np.zeros_like(sum_dict[chrom])
        prob[mask] = sum_dict[chrom][mask]/cnt[mask]

        i=0
        while i<length:
            if not mask[i]:
                i+=1; continue
            j=i
            while j<length and mask[j]:
                j+=1
            bw_out.addEntries(
                [chrom]*(j-i),
                list(range(i,j)),
                ends   = list(range(i+1,j+1)),
                values = prob[i:j].tolist()
            )
            bar.update(j-i)
            i=j

bw_out.close(); bw_a.close(); fa.close()
print("✓ BigWig готов:", args.outbw)
