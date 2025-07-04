"""
metrics.py
----------
AUROC, AUPRC, MCC и Balanced Accuracy (с fallback-ом, если
torchmetrics старой версии).

Все метрики принимают логиты любой формы, сами делают sigmoid
и flatten.
"""

import torch
from torchmetrics import Metric
from torchmetrics.classification import (
    BinaryAUROC, BinaryAveragePrecision, BinaryMatthewsCorrCoef
)

# ----- try import BalancedAccuracy ----- #
try:
    from torchmetrics.classification import BinaryBalancedAccuracy
    BalancedAccClass = BinaryBalancedAccuracy
except ImportError:
    class FallbackBalancedAccuracy(Metric):
        is_differentiable = False
        higher_is_better  = True
        full_state_update = True

        def __init__(self):
            super().__init__()
            self.add_state("tp", default=torch.tensor(0.), dist_reduce_fx="sum")
            self.add_state("tn", default=torch.tensor(0.), dist_reduce_fx="sum")
            self.add_state("fp", default=torch.tensor(0.), dist_reduce_fx="sum")
            self.add_state("fn", default=torch.tensor(0.), dist_reduce_fx="sum")

        def update(self, preds: torch.Tensor, target: torch.Tensor):
            pred_bin = (preds >= 0.5).float()
            self.tp += ((pred_bin == 1) & (target == 1)).sum()
            self.tn += ((pred_bin == 0) & (target == 0)).sum()
            self.fp += ((pred_bin == 1) & (target == 0)).sum()
            self.fn += ((pred_bin == 0) & (target == 1)).sum()

        def compute(self):
            tpr = self.tp / (self.tp + self.fn + 1e-8)
            tnr = self.tn / (self.tn + self.fp + 1e-8)
            return 0.5 * (tpr + tnr)

    BalancedAccClass = FallbackBalancedAccuracy


# ------------------------------------------------------------------------- #
class MetricCollection:
    def __init__(self, names, device="cpu"):
        self.mets = {}
        for n in names:
            if n == "auroc":
                self.mets[n] = BinaryAUROC().to(device)
            elif n == "auprc":
                self.mets[n] = BinaryAveragePrecision().to(device)
            elif n == "mcc":
                self.mets[n] = BinaryMatthewsCorrCoef().to(device)
            elif n == "bacc":
                self.mets[n] = BalancedAccClass().to(device)
            else:
                raise ValueError(f"Unknown metric '{n}'")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        p = preds.detach().sigmoid().flatten()
        t = targets.detach().flatten().int()
        for m in self.mets.values():
            m.update(p, t)

    def compute(self):
        return {k: float(v.compute()) for k, v in self.mets.items()}

    def reset(self):
        for m in self.mets.values():
            m.reset()
