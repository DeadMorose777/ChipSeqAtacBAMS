import torch, torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from .base_model import BaseModel

class TransformerModel(BaseModel):
    def __init__(self, cfg):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(cfg["model_name"])
        self.backbone = AutoModel.from_pretrained(cfg["model_name"])
        h = self.backbone.config.hidden_size
        self.at_proj = nn.Linear(1,h)          # ATAC → same dim
        self.head    = nn.Linear(h,1)          # per token → logit

    def _tokenize(self, seqs):
        toks = self.tok(seqs, return_tensors="pt",
                        padding="longest", truncation=True,
                        add_special_tokens=False)
        return toks

    def forward(self, batch):
        toks = self._tokenize(batch["seq_raw"])     # keep raw for collate
        atac = batch["atac_tok"]                    # (ΣLen,1)
        emb  = self.backbone.embeddings.word_embeddings(toks["input_ids"])
        emb += self.at_proj(atac.unsqueeze(-1))     # inject ATAC
        out  = self.backbone(inputs_embeds=emb,
                             attention_mask=toks["attention_mask"]).last_hidden_state
        logits = self.head(out).squeeze(-1)         # (B,TokLen)
        # upsample до длины L (думаем seq_len≈tok_len)
        return logits

    def collate_fn(self, batch):
        seqs = [s["seq"] for s in batch]
        toks = self._tokenize(seqs)
        # средний ATAC на каждый 6-мер токен
        atac_tok = []
        for i,s in enumerate(batch):
            vec = s["atac"].squeeze().numpy()
            L = len(vec); k=6
            means = [vec[j:j+k].mean() for j in range(L-k+1)]
            atac_tok.extend(means)
        atac_tok = torch.tensor(atac_tok, dtype=torch.float32)
        return {"seq_raw":seqs,"atac_tok":atac_tok,
                "label": torch.stack([s["label"] for s in batch])}
