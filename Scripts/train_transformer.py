#!/usr/bin/env python3
"""
Файн-тюнинг DNABERT-6: среднее ATAC сквозь отдельную голову.
"""
import json, argparse, yaml, torch, numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

ap=argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--json",   required=True)
ap.add_argument("--save",   default="dbert.pt")
ap.add_argument("--device", default="cuda")
args=ap.parse_args()

cfg=yaml.safe_load(open(args.config))
tok=AutoTokenizer.from_pretrained(cfg['model_name'])
model=AutoModelForSequenceClassification.from_pretrained(
        cfg['model_name'], num_labels=1, problem_type="regression").to(args.device)
if cfg.get("freeze_backbone",True):
    for p in model.base_model.parameters(): p.requires_grad=False

class DS(torch.utils.data.Dataset):
    def __init__(self,path):
        self.recs=[json.loads(x) for x in open(path)]
    def __len__(self): return len(self.recs)
    def __getitem__(self,i):
        o=self.recs[i]
        enc=tok(o["seq"], return_tensors="pt", padding="max_length",
                truncation=True, max_length=256)
        enc={k:v.squeeze(0) for k,v in enc.items()}
        enc["labels"]=torch.tensor([o["label"]],dtype=torch.float)
        return enc
ds=DS(args.json)

ta=TrainingArguments(output_dir="tmp",
    per_device_train_batch_size=cfg['batch'],
    num_train_epochs=cfg['epochs'],
    learning_rate=cfg['lr'],
    save_strategy="no",
    logging_steps=100)

trainer=Trainer(model=model, args=ta, train_dataset=ds)
trainer.train()
model.save_pretrained("dbert_model"); tok.save_pretrained("dbert_model")
print("✓ transformer сохранён: dbert_model/")
