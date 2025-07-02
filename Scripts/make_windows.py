#!/usr/bin/env python3
"""
Создаёт dataset.jsonl: каждая строка =
{seq:str, atac:list[float], label:int, sample:str}
"""
import json, random, pathlib, pyBigWig, pyfaidx, os
from multiprocessing.pool import ThreadPool

ROOT    = pathlib.Path(__file__).resolve().parents[1]
GENOME  = os.environ.get("GENOME", "/mnt/d/ref/hg38.fa")
WINDOW  = int(os.environ.get("WINDOW", 500))
NEG_MULT= int(os.environ.get("NEG", 3))

fa   = pyfaidx.Fasta(GENOME)
peaks= list((ROOT/"peaks").glob("*/*_summits.bed"))
bw_atac = {p.stem.split("_")[0]:str(p) for p in (ROOT/"bw").glob("*_ATAC.bw")}

def one_sample(pbed):
    sample = pbed.parent.name
    atacbw = pyBigWig.open(bw_atac[sample])
    out=[]
    with open(pbed) as bed:
        for ln in bed:
            chr,s,e,*_=ln.split(); s,e=int(s),int(e)
            mid=(s+e)//2; a=mid-WINDOW//2; b=mid+WINDOW//2
            seq = fa[chr][a:b].seq.upper()
            sig = atacbw.values(chr,a,b)
            out.append({"seq":seq,"atac":sig,"label":1,"sample":sample})
            for _ in range(NEG_MULT):
                st=random.randint(0,len(fa[chr])-WINDOW)
                seqN=fa[chr][st:st+WINDOW].seq.upper()
                sigN=atacbw.values(chr,st,st+WINDOW)
                out.append({"seq":seqN,"atac":sigN,"label":0,"sample":sample})
    return out

records=[]
with ThreadPool(8) as pool:
    for chunk in pool.imap_unordered(one_sample, peaks):
        records.extend(chunk)

with open(ROOT/"dataset.jsonl","w") as fw:
    for r in records: fw.write(json.dumps(r)+"\n")
print("✓ windows:", len(records))
