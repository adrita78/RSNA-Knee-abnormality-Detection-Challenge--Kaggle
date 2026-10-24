"""LEVER F: cheap per-plane fusion — does the cross-plane information that
Adrita's BiLSTM reached (+0.024 inside her architecture) survive without the
recurrent encoder that cost -0.030?

Her result showed plane fusion is not inherently null; mean-pool CONCATENATION
just does not reach it. So try fusions with a handful of parameters instead of
a sequence model.

Honest target, stated before running: beat the best SINGLE plane (axial,
0.7514) by >= 0.02, the pre-registered threshold. Mean-pool concatenation
already gives 0.7627, i.e. +0.011, which fails it.

    F0  axial alone                      reference, 0.7514
    F1  concat of 3 mean-pooled planes   reference, 0.7627
    F2  softmax-weighted SUM of planes   3 learned scalars
    F3  per-plane element-wise gate      3 x 768 learned weights
Both F2/F3 keep the head at 768-d, so they add ~3 or ~2.3k parameters rather
than tripling the head input.
"""
import os, sys, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
torch.set_num_threads(1)

D = ("/media/hyunsang/Ultra Touch/rsna-knee/kagglehub/datasets/alenic/"
     "rsna-knee-abnormality-detection-jpeg-224x224/versions/1")
DIRS = {"sagittal": "weak/feat", "coronal": "weak/feat_coronal", "axial": "weak/feat_axial"}
PLANES = ["sagittal", "coronal", "axial"]
TARGETS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA",
           "Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
EPOCHS, BS, LR, WD = 80, 16, 3e-4, 1e-4

t = pd.read_csv(f"{D}/train.csv"); s = pd.read_csv(f"{D}/train_series.csv")
sfs = s[(s["Anatomical_Plane"].str.lower()=="sagittal")
        & (s["Fluid_Sensitive"]==1) & (s["Fat_Suppression"]==1)]
one = (sfs.sort_values(["StudyInstanceUID","SeriesInstanceUID"])
          .drop_duplicates("StudyInstanceUID", keep="first"))
model_df = t.merge(one[["StudyInstanceUID","SeriesInstanceUID"]], on="StudyInstanceUID",
                   how="inner").reset_index(drop=True)
clean = model_df.loc[model_df[TARGETS].notna().all(axis=1)].reset_index(drop=True)
weak = pd.read_csv("weak/weak_labels_by_study.csv")
weak = weak[weak["StudyInstanceUID"].isin(model_df["StudyInstanceUID"])].reset_index(drop=True)
assert len(clean)==56 and len(weak)==1720

def pooled(sid, plane):
    p=f"{DIRS[plane]}/{sid}.npy"
    if not os.path.exists(p): return None
    return np.load(p).astype(np.float32).reshape(-1,768).mean(0)

def stack(sids):
    """(N, 3, 768) with a zero vector for an absent plane, plus (N,3) presence."""
    X=np.zeros((len(sids),3,768),np.float32); M=np.zeros((len(sids),3),np.float32)
    for i,sid in enumerate(sids):
        for j,pl in enumerate(PLANES):
            v=pooled(sid,pl)
            if v is not None: X[i,j]=v; M[i,j]=1.0
    return torch.tensor(X), torch.tensor(M)

Xc,Mc = stack(clean["StudyInstanceUID"]); Xw,Mw = stack(weak["StudyInstanceUID"])
Yc = clean[TARGETS].to_numpy(np.float32)
Yw = torch.tensor(weak[TARGETS].to_numpy(np.float32))
print(f"clean {tuple(Xc.shape)} weak {tuple(Xw.shape)}  "
      f"plane presence {Mc.mean(0).tolist()} / {Mw.mean(0).tolist()}", flush=True)

def head(dim, seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,256), nn.GELU(),
                         nn.Dropout(0.3), nn.Linear(256,12))

class Fuse(nn.Module):
    def __init__(self, mode, seed):
        super().__init__(); torch.manual_seed(seed)
        self.mode = mode
        if mode == "axial":   self.head = head(768, seed)
        elif mode == "concat":self.head = head(3*768, seed)
        elif mode == "scalar":
            self.w = nn.Parameter(torch.zeros(3)); self.head = head(768, seed)
        elif mode == "gate":
            self.w = nn.Parameter(torch.zeros(3,768)); self.head = head(768, seed)
    def forward(self, X, M):                       # X (B,3,768)  M (B,3)
        if self.mode == "axial":  return self.head(X[:,2])
        if self.mode == "concat": return self.head(X.reshape(X.shape[0],-1))
        if self.mode == "scalar":
            a = self.w.unsqueeze(0).masked_fill(M == 0, float("-inf"))
            a = torch.softmax(a, 1).unsqueeze(-1)
            return self.head((X*a).sum(1))
        a = torch.softmax(self.w.unsqueeze(0).expand(X.shape[0],-1,-1)
                          .masked_fill((M == 0).unsqueeze(-1), float("-inf")), 1)
        return self.head((X*a).sum(1))

def macro_auc(yt,yp):
    a=[]
    for i in range(12):
        ok=np.isfinite(yt[:,i])&np.isfinite(yp[:,i]); y,p=yt[ok,i],yp[ok,i]
        a.append(roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan)
    f=[x for x in a if np.isfinite(x)]
    return (float(np.mean(f)) if f else np.nan), a

def pw_of(Y):
    w=[]
    for i in range(Y.shape[1]):
        v=Y[:,i].numpy(); p,n=int((v==1).sum()),int((v==0).sum())
        w.append(1.0 if (p==0 or n==0) else min(n/p,20.0))
    return torch.tensor(w,dtype=torch.float32)

def cv(mode, seed, n_splits=5):
    strata=pd.qcut(pd.Series(Yc.sum(1)).rank(method="first"),q=n_splits,labels=False)
    skf=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=20261020)
    oof=np.full_like(Yc,np.nan,dtype=np.float64)
    for tr,ev in skf.split(np.zeros(len(Yc)),strata):
        X=torch.cat([Xc[tr],Xw]); M=torch.cat([Mc[tr],Mw])
        Y=torch.cat([torch.tensor(Yc[tr]),Yw]); pw=pw_of(Y)
        net=Fuse(mode,seed); opt=torch.optim.AdamW(net.parameters(),lr=LR,weight_decay=WD)
        g=torch.Generator().manual_seed(seed+1000)
        for _ in range(EPOCHS):
            net.train(); perm=torch.randperm(len(X),generator=g)
            for i in range(0,len(X),BS):
                j=perm[i:i+BS]
                el=F.binary_cross_entropy_with_logits(net(X[j],M[j]),Y[j],
                                                      pos_weight=pw,reduction="none")
                loss=el.mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): oof[ev]=torch.sigmoid(net(Xc[ev],Mc[ev])).numpy().astype(np.float64)
    return macro_auc(Yc.astype(np.float64),oof)

if __name__ == "__main__":
    arms=[("F0 axial alone","axial"),("F1 concat 3 planes","concat"),
          ("F2 softmax-weighted sum","scalar"),("F3 per-plane gate","gate")]
    rows=[]
    for label,mode in arms:
        res=[cv(mode,s) for s in (0,1,2)]
        a=[r[0] for r in res]
        rows.append(dict(arm=label,mode=mode,mean=float(np.mean(a)),sd=float(np.std(a)),
                         aucs=[round(x,4) for x in a]))
        print(f"{label:26s} {np.mean(a):.4f} +- {np.std(a):.4f}  {rows[-1]['aucs']}",flush=True)
        json.dump(rows,open("weak/lever_f_fusion.json","w"),indent=1)
    base=[r for r in rows if r["mode"]=="axial"][0]["mean"]
    print(f"\nvs best single plane ({base:.4f}), threshold +0.02:")
    for r in rows:
        if r["mode"]!="axial":
            d=r["mean"]-base
            print(f"  {r['arm']:26s} {d:+.4f}  {'PASSES' if d>=0.02 else 'fails'}")
    print("DONE")
