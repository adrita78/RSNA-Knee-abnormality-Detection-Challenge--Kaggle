"""LEVER B (PLAN_08 Amendment 5): add the other planes.

Only sagittal_fs is used today. Coronal_fs and axial_fs features are now
cached for the 1,778 studies we train on. Concatenate study-level vectors
across planes.

MISSING-PLANE HANDLING (pre-registered): 1,706 of 1,778 studies have a
coronal_fs series; all 1,778 have axial_fs. A missing plane contributes a
ZERO vector PLUS a binary presence flag, so the model can tell "this plane's
features are zero" from "this plane is absent". Silently dropping those 72
studies would change the training set between arms and make the comparison
dishonest.

Aggregator per plane is passed on the command line and should be whatever
Lever A selected. Evaluation is unchanged from Amendment 4.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, os, sys

torch.set_num_threads(1)   # see memory: 50x slowdown from oversubscription

D = ("/media/hyunsang/Ultra Touch/rsna-knee/kagglehub/datasets/alenic/"
     "rsna-knee-abnormality-detection-jpeg-224x224/versions/1")
TARGETS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA",
           "Lateral OA","PF OA","Effusion","Synovitis","Baker's",
           "Contusion","Fracture"]
EPOCHS, BS, LR, WD = 80, 16, 3e-4, 1e-4
PLANE_DIR = {"sagittal": "weak/feat", "coronal": "weak/feat_coronal",
             "axial": "weak/feat_axial"}

AGG = sys.argv[1] if len(sys.argv) > 1 else "mean"      # mean | meanmax
ARMS = sys.argv[2:] or ["B0", "B1", "B2"]
PLANES_FOR = {"B0": ["sagittal"],
              "B1": ["sagittal", "axial"],
              "B2": ["sagittal", "coronal", "axial"],
              # single-plane diagnostics: does a plane carry a target's
              # signal on its own, or does adding it only dilute?
              "Bc": ["coronal"],
              "Ba": ["axial"],
              "Bca": ["coronal", "axial"]}

t = pd.read_csv(f"{D}/train.csv"); s = pd.read_csv(f"{D}/train_series.csv")
sfs = s[(s["Anatomical_Plane"].str.lower()=="sagittal")
        & (s["Fluid_Sensitive"]==1) & (s["Fat_Suppression"]==1)]
one = (sfs.sort_values(["StudyInstanceUID","SeriesInstanceUID"])
          .drop_duplicates("StudyInstanceUID", keep="first"))
model_df = t.merge(one[["StudyInstanceUID","SeriesInstanceUID"]],
                   on="StudyInstanceUID", how="inner").reset_index(drop=True)
clean = model_df.loc[model_df[TARGETS].notna().all(axis=1)].reset_index(drop=True)
weak = pd.read_csv("weak/weak_labels_by_study.csv")
weak = weak[weak["StudyInstanceUID"].isin(model_df["StudyInstanceUID"])].reset_index(drop=True)
assert len(clean) == 56 and len(weak) == 1720
assert not set(weak["StudyInstanceUID"]) & set(clean["StudyInstanceUID"])

def agg_vec(sid, plane):
    """(vector, present). Zero vector + flag 0 when the plane is absent."""
    dim = 768 if AGG == "mean" else 1536
    p = f"{PLANE_DIR[plane]}/{sid}.npy"
    if not os.path.exists(p):
        return np.zeros(dim, np.float32), 0.0
    a = np.load(p).astype(np.float32).reshape(-1, 768)
    v = a.mean(0) if AGG == "mean" else np.concatenate([a.mean(0), a.max(0)])
    return v.astype(np.float32), 1.0

NWEAK = os.environ.get("NWEAK")   # None = all; "0" = image labels only.
NWEAK = None if NWEAK is None else int(NWEAK)
NOFLAG = os.environ.get("NOFLAG") == "1"   # control only: isolates the
# effect of the presence-flag columns. NOT a configuration to adopt, because
# without flags a missing plane is indistinguishable from an all-zero one.

def build(sids, planes):
    rows = []
    for sid in sids:
        vs, flags = [], []
        for pl in planes:
            v, f = agg_vec(sid, pl)
            vs.append(v); flags.append(f)
        tail = [] if NOFLAG else [np.array(flags, np.float32)]
        rows.append(np.concatenate(vs + tail))
    return torch.tensor(np.stack(rows))

Yc = clean[TARGETS].to_numpy(np.float32)
Yw = torch.tensor(weak[TARGETS].to_numpy(np.float32))

def macro_auc(yt, yp):
    a=[]
    for i in range(12):
        ok = np.isfinite(yt[:,i]) & np.isfinite(yp[:,i])
        y,p = yt[ok,i], yp[ok,i]
        a.append(roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan)
    fin=[x for x in a if np.isfinite(x)]
    return (float(np.mean(fin)) if fin else np.nan), a

def pos_weight(Y, cap=20.0):
    w=[]
    for i in range(Y.shape[1]):
        v=Y[:,i].numpy(); p,n=int((v==1).sum()),int((v==0).sum())
        w.append(1.0 if (p==0 or n==0) else min(n/p,cap))
    return torch.tensor(w, dtype=torch.float32)

def mlp(dim, seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,256), nn.GELU(),
                         nn.Dropout(0.3), nn.Linear(256,12))

def cv(Xc, Xw, seed, n_splits=5):
    strata = pd.qcut(pd.Series(Yc.sum(1)).rank(method="first"), q=n_splits, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20261020)
    oof = np.full_like(Yc, np.nan, dtype=np.float64)
    for tr, ev in skf.split(np.zeros(len(Yc)), strata):
        nw = len(Xw) if NWEAK is None else min(NWEAK, len(Xw))
        Xtr = torch.cat([Xc[tr], Xw[:nw]]); Ytr = torch.cat([torch.tensor(Yc[tr]), Yw[:nw]])
        W = torch.ones(len(Xtr)); pw = pos_weight(Ytr)
        net = mlp(Xtr.shape[1], seed)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        gen = torch.Generator().manual_seed(seed+1000)
        for _ in range(EPOCHS):
            net.train()
            perm = torch.randperm(len(Xtr), generator=gen)
            for i in range(0,len(Xtr),BS):
                j = perm[i:i+BS]
                el = F.binary_cross_entropy_with_logits(
                    net(Xtr[j]), Ytr[j], pos_weight=pw, reduction="none")
                loss = (el*W[j,None]).sum()/W[j].sum().clamp(min=1.0)/12
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            oof[ev] = torch.sigmoid(net(Xc[ev])).numpy().astype(np.float64)
    return macro_auc(Yc.astype(np.float64), oof)

if __name__ == "__main__":
    print(f"aggregator={AGG}  arms={ARMS}", flush=True)
    for pl in ("coronal","axial"):
        n = sum(os.path.exists(f"{PLANE_DIR[pl]}/{x}.npy")
                for x in list(clean['StudyInstanceUID'])+list(weak['StudyInstanceUID']))
        print(f"  {pl}: {n}/{len(clean)+len(weak)} studies have cached features", flush=True)
    rows = []
    for arm in ARMS:
        planes = PLANES_FOR[arm]
        Xc = build(clean["StudyInstanceUID"], planes)
        Xw = build(weak["StudyInstanceUID"], planes)
        res = [cv(Xc, Xw, s) for s in (0,1,2)]
        aucs = [r[0] for r in res]
        per = np.nanmean([r[1] for r in res], axis=0)
        rows.append(dict(arm=arm, planes=planes, dim=int(Xc.shape[1]),
                         mean=float(np.mean(aucs)), sd=float(np.std(aucs)),
                         aucs=[round(a,4) for a in aucs],
                         per_target={TARGETS[i]: round(float(per[i]),4) for i in range(12)}))
        print(f"{arm} {planes} dim={Xc.shape[1]}: "
              f"{np.mean(aucs):.4f} +- {np.std(aucs):.4f}  {rows[-1]['aucs']}", flush=True)
        json.dump(rows, open(f"weak/lever_b_{AGG}_{'-'.join(ARMS)}{'_noflag' if NOFLAG else ''}{f'_nweak{NWEAK}' if NWEAK is not None else ''}.json","w"), indent=1)
    print("DONE")
