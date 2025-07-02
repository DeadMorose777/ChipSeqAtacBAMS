#!/usr/bin/env python3
"""
Окно sliding-window → BigWig с предсказанным FE
"""
import argparse, pyBigWig, pyfaidx, torch, json, pathlib
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ap=argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--fasta", required=True)
ap.add_argument("--atac",  required=True)
ap.add_argument("--outbw", required=True)
ap.add_argument("--window", type=int, default=500)
ap.add_argument("--step",   type=int, default=50)
args=ap.parse_args()

tok=AutoTokenizer.from_pretrained(args.model)
mdl=AutoModelForSequenceClassification.from_pretrained(args.model).eval().cuda()

fa=pyfaidx.Fasta(args.fasta)
atac=pyBigWig.open(args.atac)
bw=pyBigWig.open(args.outbw,"w")
chroms={c:len(fa[c]) for c in fa.keys()}
bw.addHeader(list(chroms.items()))

def score(chr,start,end):
    seq=fa[chr][start:end].seq.upper()
    enc=tok(seq,return_tensors="pt",padding="max_length",truncation=True,max_length=256)
    with torch.no_grad():
        y=mdl(**{k:v.cuda() for k,v in enc.items()}).logits.sigmoid().item()
    return y

for chr,length in chroms.items():
    vals=[]
    for s in range(0,length-args.window,args.step):
        e=s+args.window
        y=score(chr,s,e)
        vals.append((chr,s,e,y))
    bw.addEntries([v[0] for v in vals],
                  [v[1] for v in vals],
                  ends=[v[2] for v in vals],
                  values=[v[3] for v in vals])
bw.close()
print("✓ BigWig создан:", args.outbw)
