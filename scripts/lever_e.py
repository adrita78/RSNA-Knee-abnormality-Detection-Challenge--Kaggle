"""Does FROZEN FEATURE QUALITY move the number? (axial plane, mean pooling)

Same protocol as every other arm: 5-fold CV over the 56, out-of-fold pooled,
3 seeds, fixed 80 epochs. Run both with and without the weak labels, because
the 2x2 showed the two interventions partially substitute.
"""
import os, sys, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
torch.set_num_threads(1)

D = ("/media/hyunsang/Ultra Touch/rsna-knee/kagglehub/datasets/alenic/"
     "rsna-knee-abnormality-detection-jpeg-224x224/versions/1")
TARGETS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA",
           "Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
BACKBONES = {"convnext_small": ("weak/feat_axial", 768),
             "resnet50": ("weak/feat_axial_resnet50", 2048),
             "convnext_base": ("weak/feat_axial_convnext_base", 1024)}
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
Yc = clean[TARGETS].to_numpy(np.float32); Yw = torch.tensor(weak[TARGETS].to_numpy(np.float32))

def macro_auc(yt, yp):
    a=[]
    for i in range(12):
        ok=np.isfinite(yt[:,i])&np.isfinite(yp[:,i]); y,p=yt[ok,i],yp[ok,i]
        a.append(roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan)
    f=[x for x in a if np.isfinite(x)]
    return (float(np.mean(f)) if f else np.nan), a

def pos_weight(Y, cap=20.0):
    w=[]
    for i in range(Y.shape[1]):
        p,n=int((Y[:,i]==1).sum()),int((Y[:,i]==0).sum())
        w.append(1.0 if (p==0 or n==0) else min(n/p,cap))
    return torch.tensor(w, dtype=torch.float32)

def load(dirname, dim, sids):
    out=[]
    for sid in sids:
        p=f"{dirname}/{sid}.npy"
        a=np.load(p).astype(np.float32).reshape(-1, dim)
        out.append(a.mean(0))
    return torch.tensor(np.stack(out))

def cv(Xc, Xw, n_weak, seed, n_splits=5):
    strata = pd.qcut(pd.Series(Yc.sum(1)).rank(method="first"), q=n_splits, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20261020)
    oof = np.full_like(Yc, np.nan, dtype=np.float64)
    for tr, ev in skf.split(np.zeros(len(Yc)), strata):
        X = torch.cat([Xc[tr], Xw[:n_weak]]); Y = torch.cat([torch.tensor(Yc[tr]), Yw[:n_weak]])
        pw = pos_weight(Y)
        torch.manual_seed(seed)
        net = nn.Sequential(nn.LayerNorm(X.shape[1]), nn.Linear(X.shape[1],256), nn.GELU(),
                            nn.Dropout(0.3), nn.Linear(256,12))
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        g = torch.Generator().manual_seed(seed+1000)
        for _ in range(EPOCHS):
            net.train()
            perm = torch.randperm(len(X), generator=g)
            for i in range(0,len(X),BS):
                j=perm[i:i+BS]
                el=F.binary_cross_entropy_with_logits(net(X[j]), Y[j], pos_weight=pw,
                                                      reduction="none")
                loss=el.mean(); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): oof[ev]=torch.sigmoid(net(Xc[ev])).numpy().astype(np.float64)
    return macro_auc(Yc.astype(np.float64), oof)

if __name__ == "__main__":
    rows=[]
    for name in (sys.argv[1:] or list(BACKBONES)):
        dirname, dim = BACKBONES[name]
        if not os.path.isdir(dirname):
            print(f"{name}: no cache yet, skipping"); continue
        have = {f[:-4] for f in os.listdir(dirname)}
        if not set(clean["StudyInstanceUID"]) <= have:
            print(f"{name}: incomplete cache ({len(have)}), skipping"); continue
        Xc = load(dirname, dim, clean["StudyInstanceUID"])
        Xw = load(dirname, dim, weak["StudyInstanceUID"])
        for n_weak, tag in ((0, "image labels only"), (len(Xw), "+1,720 weak")):
            a=[cv(Xc, Xw, n_weak, s)[0] for s in (0,1,2)]
            rows.append(dict(backbone=name, dim=dim, n_weak=n_weak,
                             mean=float(np.mean(a)), sd=float(np.std(a)),
                             aucs=[round(x,4) for x in a]))
            print(f"{name:15s} ({dim:4d}-d) {tag:18s}: {np.mean(a):.4f} +- {np.std(a):.4f}  "
                  f"{rows[-1]['aucs']}", flush=True)
        json.dump(rows, open("weak/lever_e_backbones.json","w"), indent=1)
    print("\nconvnext_small refs (axial, no flag): 0.6446 no-weak | 0.7514 +weak")
    print("DONE")
