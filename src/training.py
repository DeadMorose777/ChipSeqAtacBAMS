"""
Общий цикл обучения + логирование:

• метрики каждой эпохи пишутся в TensorBoard (runs/.../tb/)
• по окончании обучения сохраняются PNG-кривые (auroc_curve.png, …)
• history.json хранит массивы всех значений
"""

import os, json, yaml
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt               # <- автографики

from .metrics import MetricCollection
from .registry import get_model_cls
from .dataset import build_loaders


# --------------------------------------------------------------------------- #
def fit(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    save_dir = Path(cfg["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(save_dir / "run_config.json", "w"), indent=2)

    # ---------- модель ----------------------------------------------------- #
    model_cfg = yaml.safe_load(open(cfg["model_cfg"]))
    model_name = Path(cfg["model_cfg"]).stem
    ModelCls = get_model_cls(model_name)
    model = ModelCls(model_cfg).to(cfg["device"])

    # ---------- данные ------------------------------------------------------ #
    loaders = build_loaders(cfg, model.collate_fn)
    metrics = MetricCollection(cfg["metrics"], cfg["device"])

    # ---------- оптимизатор ------------------------------------------------- #
    optim_cls = getattr(torch.optim, cfg["optim"]["name"])
    lr = float(cfg["optim"]["lr"])
    wd = float(cfg["optim"]["weight_decay"])
    optim = optim_cls(model.parameters(), lr=lr, weight_decay=wd)

    # ---------- TensorBoard ------------------------------------------------- #
    tb = SummaryWriter(log_dir=str(save_dir / "tb"))

    # ---------- для PNG-кривых --------------------------------------------- #
    history = defaultdict(lambda: defaultdict(list))  # history[phase][metric]

    best_val = 0.0
    for epoch in range(1, cfg["trainer"]["epochs"] + 1):
        train_scores = _run_one_epoch(
            model, loaders[0], optim, metrics, cfg, epoch, "train", tb
        )
        val_scores = _run_one_epoch(
            model, loaders[1], None, metrics, cfg, epoch, "val", tb
        )

        for k, v in train_scores.items():
            history["train"][k].append(v)
            history["val"][k].append(val_scores[k])
        history["epoch"].append(epoch)

        # — сохраняем лучшую по AUPRC модель —
        if val_scores.get("auprc", 0.0) > best_val:
            best_val = val_scores["auprc"]
            torch.save(model.state_dict(), save_dir / "best.pt")

    # ---------- test -------------------------------------------------------- #
    model.load_state_dict(torch.load(save_dir / "best.pt"))
    _run_one_epoch(
        model, loaders[2], None, metrics, cfg, cfg["trainer"]["epochs"] + 1, "test", tb
    )
    tb.close()

    # ---------- сохраняем историю и PNG-кривые ----------------------------- #
    json.dump(history, open(save_dir / "history.json", "w"), indent=2)
    _plot_curves(save_dir, history)
    print("✓ Графики сохранены в", save_dir)


# --------------------------------------------------------------------------- #
def _run_one_epoch(
    model, loader, optim, metrics, cfg, epoch, phase: str, tb: SummaryWriter
):
    model.train(phase == "train")
    metrics.reset()

    pbar = tqdm(loader, desc=f"{phase.upper()} {epoch}", leave=False)
    for batch in pbar:
        # перенос на GPU/CPU
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(cfg["device"])

        preds = model(batch)
        targets = batch["label"].to(cfg["device"])

        if optim:  # --- train ---
            loss = torch.nn.BCELoss()(preds, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()

        # --- метрики ---
        metrics.update(preds, targets.int())

    scores = metrics.compute()
    pbar.close()
    print(
        f"{phase.upper()} E{epoch}: "
        + " ".join(f"{k}={v:.4f}" for k, v in scores.items())
    )

    # --- TensorBoard ---
    for k, v in scores.items():
        tb.add_scalar(f"{phase}/{k}", v, epoch)

    return scores


# --------------------------------------------------------------------------- #
def _plot_curves(save_dir: Path, hist):
    """
    Сохраняет PNG для каждой метрики: train vs val.
    """
    epochs = hist["epoch"]
    for metric in hist["train"]:
        plt.figure()
        plt.plot(epochs, hist["train"][metric], label="train")
        plt.plot(epochs, hist["val"][metric], label="val")
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"{metric}_curve.png", dpi=120)
        plt.close()
