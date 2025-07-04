"""
training.py
===========

• вход/выход моделей — (B, L)   (логиты + бинарная маска)
• loss            — BCEWithLogitsLoss
• метрики         — AUROC, AUPRC, MCC, balanced accuracy
• логирование     — TensorBoard (+ PNG-кривые в конце)
"""

import json, yaml
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from .metrics  import MetricCollection
from .registry import get_model_cls
from .dataset  import build_loaders


# --------------------------------------------------------------------------- #
def fit(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    run_dir = Path(cfg["save_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(run_dir / "run_config.json", "w"), indent=2)

    # ---------- модель ----------------------------------------------------- #
    model_cfg = yaml.safe_load(open(cfg["model_cfg"]))
    ModelCls  = get_model_cls(Path(cfg["model_cfg"]).stem)
    model     = ModelCls(model_cfg).to(cfg["device"])

    # ---------- данные ------------------------------------------------------ #
    loaders  = build_loaders(cfg, model.collate_fn)
    metrics  = MetricCollection(cfg["metrics"], cfg["device"])

    # ---------- оптимизатор ------------------------------------------------- #
    opt_cls = getattr(torch.optim, cfg["optim"]["name"])
    optim   = opt_cls(model.parameters(),
                      lr=float(cfg["optim"]["lr"]),
                      weight_decay=float(cfg["optim"]["weight_decay"]))

    # ---------- TensorBoard ------------------------------------------------- #
    tb = SummaryWriter(str(run_dir / "tb"))

    # ---------- история для PNG-кривых ------------------------------------- #
    history = {"epoch": [], "train": defaultdict(list), "val": defaultdict(list)}

    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_val = 0.0
    for epoch in range(1, cfg["trainer"]["epochs"] + 1):
        train_scores = _run_epoch(model, loaders[0], optim,  loss_fn,
                                  metrics, cfg, epoch, "train", tb)
        val_scores   = _run_epoch(model, loaders[1], None,   loss_fn,
                                  metrics, cfg, epoch, "val",   tb)

        history["epoch"].append(epoch)
        for k, v in train_scores.items():
            history["train"][k].append(v)
            history["val"][k].append(val_scores[k])

        # — сохраняем лучшую по AUPRC —
        if val_scores.get("auprc", 0.0) > best_val:
            best_val = val_scores["auprc"]
            torch.save(model.state_dict(), run_dir / "best.pt")

    # ---------- test -------------------------------------------------------- #
    model.load_state_dict(torch.load(run_dir / "best.pt"))
    _run_epoch(model, loaders[2], None, loss_fn,
               metrics, cfg, epoch+1, "test", tb)
    tb.close()

    # ---------- сохранить историю + PNG-кривые ----------------------------- #
    json.dump(history, open(run_dir / "history.json", "w"), indent=2)
    _plot_curves(run_dir, history)
    print("✓ Графики сохранены в", run_dir)


# --------------------------------------------------------------------------- #
def _run_epoch(model, loader, optim, loss_fn, metrics,
               cfg, epoch, phase, tb):
    model.train(phase == "train")
    metrics.reset()

    pbar = tqdm(loader, desc=f"{phase.upper()} {epoch}", leave=False)
    for batch in pbar:
        # to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(cfg["device"])

        preds   = model(batch)                  # (B, L) logits
        targets = batch["label"]                # (B, L)

        if optim is not None:                   # TRAIN
            loss = loss_fn(preds, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()

        metrics.update(preds, targets)

    scores = metrics.compute()
    pbar.close()
    print(f"{phase.upper()} E{epoch}: " +
          " ".join(f"{k}={v:.4f}" for k, v in scores.items()))

    # TensorBoard
    for k, v in scores.items():
        tb.add_scalar(f"{phase}/{k}", v, epoch)

    return scores


# --------------------------------------------------------------------------- #
def _plot_curves(run_dir: Path, hist):
    """PNG-кривые train/val для каждой метрики."""
    epochs = hist["epoch"]
    for metric in hist["train"]:
        plt.figure()
        plt.plot(epochs, hist["train"][metric], label="train")
        plt.plot(epochs, hist["val"][metric],   label="val")
        plt.xlabel("epoch");  plt.ylabel(metric);  plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / f"{metric}_curve.png", dpi=120)
        plt.close()
