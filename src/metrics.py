from torchmetrics.classification import (
    BinaryAUROC, BinaryAveragePrecision, BinaryF1Score,
    BinaryAccuracy
)
class MetricCollection:
    def __init__(self, names: list[str], device: str = "cpu"):
        self.mets = {}
        for n in names:
            if n == "auroc":  self.mets[n] = BinaryAUROC().to(device)
            if n == "auprc":  self.mets[n] = BinaryAveragePrecision().to(device)
            if n == "f1":     self.mets[n] = BinaryF1Score().to(device)
            if n == "acc":    self.mets[n] = BinaryAccuracy().to(device)

    def update(self, preds, targets):
        for m in self.mets.values():
            m.update(preds, targets)

    def compute(self) -> dict[str, float]:
        return {k: float(m.compute()) for k, m in self.mets.items()}

    def reset(self):
        for m in self.mets.values():
            m.reset()
