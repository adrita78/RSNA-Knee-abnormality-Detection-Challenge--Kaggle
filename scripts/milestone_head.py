"""Milestone (pre-registered in PLAN_08.md): head-only training on cached
frozen-ConvNeXt features; scale the number of Claude soft-labeled studies
and measure the val-AUC slope on the untouched 12 (seed-42 split).
Decision rule: if val AUC does not move toward 0.8 with more soft labels,
stop labeling and report; otherwise continue to 3,794.
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import itertools, json, sys

SEED_SPLIT = 42
D = "/media/hyunsang/Ultra Touch/rsna-knee/kagglehub/datasets/alenic/rsna-knee-abnormality-detection-jpeg-224x224/versions/1"
FEAT = "weak/feat"
TARGETS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA",
           "Lateral OA","PF OA","Effusion","Synovitis","Baker's",
           "Contusion","Fracture"]

# ---- reproduce the notebook split exactly (asserted vs prepare_subset) ----
t = pd.read_csv(f"{D}/train.csv"); s = pd.read_csv(f"{D}/train_series.csv")
sfs = s[(s["Anatomical_Plane"].str.lower()=="sagittal") & (s["Fluid_Sensitive"]==1) & (s["Fat_Suppression"]==1)]
one = sfs.sort_values(["StudyInstanceUID","SeriesInstanceUID"]).drop_duplicates("StudyInstanceUID", keep="first")
model_df = t.merge(one[["StudyInstanceUID","SeriesInstanceUID"]], on="StudyInstanceUID", how="inner").reset_index(drop=True)
complete_mask = model_df[TARGETS].notna().all(axis=1)
assert len(model_df) == 4150 and int(complete_mask.sum()) == 56
complete_df = model_df.loc[complete_mask].copy()
strata = complete_df[TARGETS].sum(axis=1).astype(int)
if strata.value_counts().min() < 2:
    strata = pd.qcut(strata.rank(method="first"), q=5, labels=False)
ctrain, val_data = train_test_split(complete_df, test_size=0.20,
                                    random_state=SEED_SPLIT, stratify=strata)
assert len(ctrain) == 44 and len(val_data) == 12

# ---- soft labels: sample300 then remaining, in idx order -----------------
def load_soft(label_csv, report_csv):
    lab = pd.read_csv(label_csv); rep = pd.read_csv(report_csv)
    lab = lab.sort_values("idx").reset_index(drop=True)
    sids = rep["StudyInstanceUID"].iloc[lab["idx"].values].values
    out = lab[ [c for c in lab.columns if c != "idx"] ].copy()
    out.columns = TARGETS
    out.insert(0, "StudyInstanceUID", sids)
    return out

soft = pd.concat([load_soft("weak/claude_labels_300.csv", "weak/sample300.csv"),
                  load_soft("weak/claude_labels_remaining.csv", "weak/remaining.csv")],
                 ignore_index=True)
valset = set(val_data["StudyInstanceUID"])
assert not (set(soft["StudyInstanceUID"]) & valset), "soft labels leak into val"
assert not (set(soft["StudyInstanceUID"]) & set(ctrain["StudyInstanceUID"]))
print(f"soft pool: {len(soft)} studies")

def pooled_feat(sid):
    a = np.load(f"{FEAT}/{sid}.npy").astype(np.float32).reshape(-1, 768)
    return a.mean(0)

print("loading features...", flush=True)
feat_cache = {}
for sid in itertools.chain(ctrain["StudyInstanceUID"], val_data["StudyInstanceUID"],
                           soft["StudyInstanceUID"]):
    if sid not in feat_cache:
        feat_cache[sid] = pooled_feat(sid)

Xval = torch.tensor(np.stack([feat_cache[s] for s in val_data["StudyInstanceUID"]]))
Yval = val_data[TARGETS].to_numpy(np.float64)
Xhard = torch.tensor(np.stack([feat_cache[s] for s in ctrain["StudyInstanceUID"]]))
Yhard = torch.tensor(ctrain[TARGETS].to_numpy(np.float32))
Xsoft_all = torch.tensor(np.stack([feat_cache[s] for s in soft["StudyInstanceUID"]]))
Ysoft_all = torch.tensor(soft[TARGETS].to_numpy(np.float32))

def macro_auc(y_true, y_pred):
    aucs = []
    for i in range(len(TARGETS)):
        valid = np.isfinite(y_true[:, i]) & np.isfinite(y_pred[:, i])
        yt, yp = y_true[valid, i], y_pred[valid, i]
        aucs.append(roc_auc_score(yt, yp)
                    if len(yt) and len(np.unique(yt)) == 2 else np.nan)
    finite = [a for a in aucs if np.isfinite(a)]
    return (float(np.mean(finite)) if finite else np.nan), aucs

def pos_weight_from(Y, max_weight=20.0):
    ws = []
    for i in range(Y.shape[1]):
        v = Y[:, i].numpy()
        pos, neg = int((v == 1).sum()), int((v == 0).sum())
        ws.append(1.0 if (pos == 0 or neg == 0) else min(neg / pos, max_weight))
    return torch.tensor(ws, dtype=torch.float32)

def make_head(seed):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    return nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(),
                         nn.Dropout(0.3), nn.Linear(256, 12))

def run(n_soft, w_soft, seed, epochs=80, bs=16, lr=3e-4, wd=1e-4):
    X = torch.cat([Xhard, Xsoft_all[:n_soft]])
    Y = torch.cat([Yhard, Ysoft_all[:n_soft]])
    W = torch.cat([torch.ones(len(Xhard)), torch.full((n_soft,), w_soft)])
    pw = pos_weight_from(Y)
    head = make_head(seed)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=wd)
    g = torch.Generator().manual_seed(seed + 1000)
    best = -1.0; best_ep = -1
    for ep in range(epochs):
        head.train()
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), bs):
            idx = perm[i:i+bs]
            logits = head(X[idx])
            el = F.binary_cross_entropy_with_logits(
                logits, Y[idx], pos_weight=pw, reduction="none")
            loss = (el * W[idx, None]).sum() / W[idx].sum().clamp(min=1.0) / 12
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            pv = torch.sigmoid(head(Xval)).numpy().astype(np.float64)
        auc, _ = macro_auc(Yval, pv)
        if np.isfinite(auc) and auc > best:
            best, best_ep = auc, ep
    return best, best_ep

if __name__ == "__main__":
    n_avail = len(Xsoft_all)
    sizes = [n for n in [0, 100, 200, 300, n_avail] if n <= n_avail]
    sizes = sorted(set(sizes))
    rows = []
    for n_soft in sizes:
        for w_soft in ([1.0] if n_soft == 0 else [1.0, 0.3]):
            aucs = [run(n_soft, w_soft, seed)[0] for seed in (0, 1, 2)]
            rows.append(dict(n_soft=n_soft, w_soft=w_soft,
                             auc_mean=float(np.mean(aucs)),
                             auc_sd=float(np.std(aucs)),
                             aucs=[round(a, 4) for a in aucs]))
            print(f"n_soft={n_soft:4d} w={w_soft}: "
                  f"{np.mean(aucs):.4f} +- {np.std(aucs):.4f}  {rows[-1]['aucs']}", flush=True)
    json.dump(rows, open("weak/milestone_results.json", "w"), indent=1)
    print("DONE")
