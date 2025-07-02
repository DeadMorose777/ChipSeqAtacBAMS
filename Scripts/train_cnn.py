#!/usr/bin/env python3
import torch, torch.nn as nn, json, argparse, yaml, numpy as np

class OneHot:
    _map={b'A':0,b'C':1,b'G':2,b'T':3}
    def __call__(self,seq):
        arr=np.zeros((4,len(seq)),dtype=np.float32)
        for i,b in enumerate(seq.encode()):
            if b in self._map: arr[self._map[b],i]=1
        return arr

class CNN(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.conv=nn.Conv1d(4,cfg['n_filters'],cfg['kernel'])
        self.pool=nn.MaxPool1d(cfg['pool'])
        L=(cfg['seq_length']-cfg['kernel']+1)//cfg['pool']
        self.fc1=nn.Linear(cfg['n_filters']*L+1,cfg['hidden'])
        self.fc2=nn.Linear(cfg['hidden'],1)
    def forward(self,x,at):
        x=self.pool(torch.relu(self.conv(x)))
        x=torch.cat([x.flatten(1),at],1)
        x=torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

def load(cfg, path):
    oh=OneHot(); data=[]
    for ln in open(path):
        o=json.loads(ln)
        x=oh(o["seq"]); at=np.array([[np.mean(o["atac"])]]).astype(np.float32)
        data.append((x,at,o["label"]))
    X=torch.tensor([d[0] for d in data]); AT=torch.tensor([d[1] for d in data])
    y=torch.tensor([d[2] for d in data]).float().unsqueeze(1)
    return X,AT,y

ap=argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--json", required=True)
ap.add_argument("--save", default="cnn.pt")
ap.add_argument("--device", default="cpu")
args=ap.parse_args()

cfg=yaml.safe_load(open(args.config))
X,AT,y=load(cfg,args.json)
device=torch.device(args.device)
X,AT,y=X.to(device),AT.to(device),y.to(device)

model=CNN(cfg).to(device); opt=torch.optim.Adam(model.parameters(), lr=cfg['lr'])
bsize=32; epochs=cfg['epochs']
for ep in range(epochs):
    perm=torch.randperm(len(y))
    for i in range(0,len(y),bsize):
        idx=perm[i:i+bsize]
        pred=model(X[idx],AT[idx])
        loss=nn.BCELoss()(pred,y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch {ep+1}/{epochs} loss={loss.item():.4f}")
torch.save(model.state_dict(), args.save)
print("✓ cnn сохранён:", args.save)
