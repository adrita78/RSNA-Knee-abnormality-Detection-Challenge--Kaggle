"""LEVER A (PLAN_08 Amendment 5): slice aggregator.

The head currently sees one mean-pooled vector per study. Knee findings are
focal, so test whether letting the peak slice contribute (A1) or letting the
model learn which slices matter (A2) beats mean pooling.

Same evaluation as Amendment 4: 5-fold CV over all 56, out-of-fold pooled,
fixed 80 epochs, 3 seeds, 1,720 report labels at w=1.0.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json, sys

# tiny 16x768 batches: thread sync dominates the arithmetic, and under
# machine contention 8 threads measured 50x SLOWER than 1. Keep it at 1.
torch.set_num_threads(1)

D = ("/media/hyunsang/Ultra Touch/rsna-knee/kagglehub/datasets/alenic/"
     "rsna-knee-abnormality-detection-jpeg-224x224/versions/1")
FEAT = "weak/feat"
TARGETS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA",
           "Lateral OA","PF OA","Effusion","Synovitis","Baker's",
           "Contusion","Fracture"]
EPOCHS, BS, LR, WD = 80, 16, 3e-4, 1e-4

t = pd.read_csv(f"{D}/train.csv"); s = pd.read_csv(f"{D}/train_series.csv")
sfs = s[(s["Anatomical_Plane"].str.lower()=="sagittal")
        & (s["Fluid_Sensitive"]==1) & (s["Fat_Suppression"]==1)]
one = (sfs.sort_values(["StudyInstanceUID","SeriesInstanceUID"])
          .drop_duplicates("StudyInstanceUID", keep="first"))
model_df = t.merge(one[["StudyInstanceUID","SeriesInstanceUID"]],
                   on="StudyInstanceUID", how="inner").reset_index(drop=True)
clean = model_df.loc[model_df[TARGETS].notna().all(axis=1)].reset_index(drop=True)
assert len(model_df) == 4150 and len(clean) == 56

weak = pd.read_csv("weak/weak_labels_by_study.csv")
weak = weak[weak["StudyInstanceUID"].isin(model_df["StudyInstanceUID"])].reset_index(drop=True)
assert not set(weak["StudyInstanceUID"]) & set(clean["StudyInstanceUID"])
assert len(weak) == 1720
print(f"clean={len(clean)} weak={len(weak)}", flush=True)

def slices(sid):
    return np.load(f"{FEAT}/{sid}.npy").astype(np.float32).reshape(-1, 768)

Sc = [torch.tensor(slices(x)) for x in clean["StudyInstanceUID"]]
Sw = [torch.tensor(slices(x)) for x in weak["StudyInstanceUID"]]
Yc = clean[TARGETS].to_numpy(np.float32)
Yw = torch.tensor(weak[TARGETS].to_numpy(np.float32))
print(f"slice tensors loaded; total slices {sum(x.shape[0] for x in Sc+Sw):,}", flush=True)

MEANc = torch.stack([x.mean(0) for x in Sc]); MEANw = torch.stack([x.mean(0) for x in Sw])
MAXc  = torch.stack([x.max(0).values for x in Sc]); MAXw = torch.stack([x.max(0).values for x in Sw])

def macro_auc(yt, yp):
    a = []
    for i in range(12):
        ok = np.isfinite(yt[:,i]) & np.isfinite(yp[:,i])
        y, p = yt[ok,i], yp[ok,i]
        a.append(roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan)
    fin = [x for x in a if np.isfinite(x)]
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

class AttnPool(nn.Module):
    """Learned query over slices; masked softmax, then the same head."""
    def __init__(self, seed, dim=768, attn_dim=128):
        super().__init__()
        torch.manual_seed(seed)
        self.score = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, attn_dim),
                                   nn.Tanh(), nn.Linear(attn_dim, 1))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim,256), nn.GELU(),
                                  nn.Dropout(0.3), nn.Linear(256,12))
    def forward(self, padded, mask):          # padded (B,S,768) mask (B,S)
        logit = self.score(padded).squeeze(-1)
        logit = logit.masked_fill(~mask, float("-inf"))
        w = torch.softmax(logit, dim=1).unsqueeze(-1)
        return self.head((padded * w).sum(1))

def pad(batch):
    S = max(x.shape[0] for x in batch)
    out = torch.zeros(len(batch), S, 768)
    msk = torch.zeros(len(batch), S, dtype=torch.bool)
    for i, x in enumerate(batch):
        out[i,:x.shape[0]] = x; msk[i,:x.shape[0]] = True
    return out, msk

def run_fold(arm, tr_idx, ev_idx, seed):
    """tr_idx indexes the clean set; weak rows are always appended."""
    Ytr = torch.cat([torch.tensor(Yc[tr_idx]), Yw])
    W   = torch.cat([torch.ones(len(tr_idx)), torch.ones(len(Yw))])
    pw  = pos_weight(Ytr)
    gen = torch.Generator().manual_seed(seed+1000)

    if arm in ("A0","A1"):
        if arm == "A0":
            Xtr = torch.cat([MEANc[tr_idx], MEANw]); Xev = MEANc[ev_idx]
        else:
            Xtr = torch.cat([torch.cat([MEANc[tr_idx], MAXc[tr_idx]],1),
                             torch.cat([MEANw, MAXw],1)])
            Xev = torch.cat([MEANc[ev_idx], MAXc[ev_idx]],1)
        net = mlp(Xtr.shape[1], seed)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
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
            return torch.sigmoid(net(Xev)).numpy().astype(np.float64)

    # A2: attention pooling over the raw slice sets
    Str = [Sc[i] for i in tr_idx] + Sw
    net = AttnPool(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
    for _ in range(EPOCHS):
        net.train()
        perm = torch.randperm(len(Str), generator=gen)
        for i in range(0,len(Str),BS):
            j = perm[i:i+BS].tolist()
            p, m = pad([Str[k] for k in j])
            el = F.binary_cross_entropy_with_logits(
                net(p,m), Ytr[j], pos_weight=pw, reduction="none")
            loss = (el*W[j,None]).sum()/W[j].sum().clamp(min=1.0)/12
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p, m = pad([Sc[i] for i in ev_idx])
        return torch.sigmoid(net(p,m)).numpy().astype(np.float64)

def cv(arm, seed, n_splits=5):
    strata = pd.qcut(pd.Series(Yc.sum(1)).rank(method="first"), q=n_splits, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20261020)
    oof = np.full_like(Yc, np.nan, dtype=np.float64)
    for tr, ev in skf.split(np.zeros(len(Yc)), strata):
        oof[ev] = run_fold(arm, tr, ev, seed)
    return macro_auc(Yc.astype(np.float64), oof)

if __name__ == "__main__":
    arms = sys.argv[1:] or ["A0","A1","A2"]
    rows = []
    for arm in arms:
        res = [cv(arm, s) for s in (0,1,2)]
        aucs = [r[0] for r in res]
        per = np.nanmean([r[1] for r in res], axis=0)
        rows.append(dict(arm=arm, mean=float(np.mean(aucs)), sd=float(np.std(aucs)),
                         aucs=[round(a,4) for a in aucs],
                         per_target={TARGETS[i]: round(float(per[i]),4) for i in range(12)}))
        print(f"{arm}: {np.mean(aucs):.4f} +- {np.std(aucs):.4f}  {rows[-1]['aucs']}", flush=True)
        json.dump(rows, open("weak/lever_a_results.json","w"), indent=1)
    print("DONE")
