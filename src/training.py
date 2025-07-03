import torch, os, yaml, json
from pathlib import Path
from tqdm import tqdm
from .metrics import MetricCollection
from .registry import get_model_cls
from .dataset import build_loaders

def fit(cfg_path: str):
    cfg = yaml.safe_load(open(cfg_path))
    save_dir = Path(cfg["save_dir"]); save_dir.mkdir(parents=True, exist_ok=True)
    json.dump(cfg, open(save_dir / "run_config.json", "w"), indent=2)

    # ---------- модель ----------
    model_cfg = yaml.safe_load(open(cfg["model_cfg"]))
    model_name = Path(cfg["model_cfg"]).stem
    ModelCls = get_model_cls(model_name)
    model = ModelCls(model_cfg).to(cfg["device"])

    # ---------- данные ----------
    loaders = build_loaders(cfg, model.collate_fn)
    metrics = MetricCollection(cfg["metrics"], cfg["device"])

    # ---------- оптимизатор ----------
    optim_cls = getattr(torch.optim, cfg["optim"]["name"])
    optim = optim_cls(model.parameters(), lr=cfg["optim"]["lr"],
                      weight_decay=cfg["optim"]["weight_decay"])

    best_val = 0.0
    for epoch in range(1, cfg["trainer"]["epochs"] + 1):
        train_one_epoch(model, loaders[0], optim, metrics, cfg, epoch, "train")
        val_score = train_one_epoch(model, loaders[1], None, metrics, cfg, epoch, "val")

        if val_score > best_val:
            best_val = val_score
            torch.save(model.state_dict(), save_dir / "best.pt")

    # ---------- test ----------
    model.load_state_dict(torch.load(save_dir / "best.pt"))
    train_one_epoch(model, loaders[2], None, metrics, cfg, "FINAL", "test")

def train_one_epoch(model, loader, optim, metrics, cfg, epoch, phase):
    model.train(phase == "train")
    metrics.reset()
    pbar = tqdm(loader, desc=f"{phase} {epoch}", leave=False)
    for batch in pbar:
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(cfg["device"])
        preds = model(batch)
        targets = batch["label"].to(cfg["device"])

        if optim:
            loss = torch.nn.BCELoss()(preds, targets)
            optim.zero_grad(); loss.backward(); optim.step()
        metrics.update(preds, targets.int())
    scores = metrics.compute()
    pbar.close()
    print(f"{phase.upper()} E{epoch}: " +
          " ".join(f"{k}={v:.4f}" for k, v in scores.items()))
    return scores.get("auprc", 0.0)   # критерий выбора best.pt
