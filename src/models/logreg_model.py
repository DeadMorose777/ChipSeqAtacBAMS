"""
Baseline логрегрессия на k-мерах (+mean ATAC) из train_logreg.py :contentReference[oaicite:4]{index=4},
но завернута в nn.Module, чтобы работала с общей тренирующей обвязкой.
"""

import itertools, torch, numpy as np, math
from sklearn.linear_model import LogisticRegression
from .base_model import BaseModel

class LogRegModel(BaseModel):
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        alpha = "ACGT"
        self.kmers = ["".join(p) for p in itertools.product(alpha, repeat=6)]
        self.clf = LogisticRegression(max_iter=300)

    # ---------- API ----------
    def forward(self, batch):
        with torch.no_grad():
            vec = batch["vec"]      # (B,4097)
            logits = torch.from_numpy(self.clf.predict_proba(vec)[:, 1])
            return logits

    def collate_fn(self, samples):
        vecs, labels = [], []
        for s in samples:
            vecs.append(self.seq2vec(s["seq"], s["atac"]))
            labels.append(s["label"])
        return {
            "vec"  : torch.tensor(np.stack(vecs), dtype=torch.float32),
            "label": torch.stack(labels),
        }

    # ---------- helpers ----------
    def seq2vec(self, seq, atac):
        d = {k:0 for k in self.kmers}
        for i in range(len(seq) - 5):
            d[seq[i:i+6]] += 1
        return np.concatenate([np.fromiter(d.values(), float), [atac.item()]])

    # ---------- fit ----------
    def sklearn_fit(self, loader):
        X, y = [], []
        for b in loader:
            for i in range(len(b["label"])):
                X.append(b["vec"][i].numpy())
                y.append(int(b["label"][i].item()))
        self.clf.fit(X, y)
