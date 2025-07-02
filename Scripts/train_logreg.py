#!/usr/bin/env python3
"""
Бейзлайн: логистическая регрессия на 6-мерах + mean(ATAC)
"""
import json, itertools, numpy as np, joblib, argparse
from sklearn.linear_model import LogisticRegression

def kmers(seq,k=6):
    alpha="ACGT"; km=["".join(p) for p in itertools.product(alpha, repeat=k)]
    d={k:0 for k in km}
    for i in range(len(seq)-k+1): d[seq[i:i+k]]+=1
    return np.fromiter(d.values(), float)

ap=argparse.ArgumentParser()
ap.add_argument("--json", required=True)
ap.add_argument("--save", default="logreg.joblib")
args=ap.parse_args()

X=[]; y=[]
for ln in open(args.json):
    o=json.loads(ln)
    vec=np.concatenate([kmers(o["seq"]), [np.mean(o["atac"])]])
    X.append(vec); y.append(o["label"])
clf=LogisticRegression(max_iter=300).fit(X,y)
joblib.dump(clf, args.save)
print("✓ logreg сохранён:", args.save)
