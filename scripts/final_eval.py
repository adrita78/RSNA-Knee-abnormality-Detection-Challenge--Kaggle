"""PLAN_08 step 5 + Amendment 4: final honest evaluation.

5-fold CV over ALL 56 clean studies. Out-of-fold predictions are pooled and
one macro AUC is computed over 56 studies. The soft (Claude report-label)
pool is disjoint from the 56 by construction, so it is reused in every fold.

PRIMARY   = fixed 80 epochs, no epoch selection on the evaluation data.
SECONDARY = best epoch by the held-out fold's AUC (the notebook protocol that
            produced 0.6559; optimistic, reported for comparability only).
"""
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json

D = "/media/hyunsang/Ultra Touch/rsna-knee/kagglehub/datasets/alenic/rsna-knee-abnormality-detection-jpeg-224x224/versions/1"
FEAT = "weak/feat"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]
EPOCHS, BS, LR, WD = 80, 16, 3e-4, 1e-4

t = pd.read_csv(f"{D}/train.csv")
s = pd.read_csv(f"{D}/train_series.csv")
sfs = s[(s["Anatomical_Plane"].str.lower() == "sagittal")
        & (s["Fluid_Sensitive"] == 1) & (s["Fat_Suppression"] == 1)]
one = (sfs.sort_values(["StudyInstanceUID", "SeriesInstanceUID"])
          .drop_duplicates("StudyInstanceUID", keep="first"))
model_df = t.merge(one[["StudyInstanceUID", "SeriesInstanceUID"]],
                   on="StudyInstanceUID", how="inner").reset_index(drop=True)
clean = model_df.loc[model_df[TARGETS].notna().all(axis=1)].copy().reset_index(drop=True)
assert len(model_df) == 4150 and len(clean) == 56


def load_soft(label_csv, report_csv):
    lab = pd.read_csv(label_csv).sort_values("idx").reset_index(drop=True)
    rep = pd.read_csv(report_csv)
    sids = rep["StudyInstanceUID"].iloc[lab["idx"].values].values
    out = lab[[c for c in lab.columns if c != "idx"]].copy()
    out.columns = TARGETS
    out.insert(0, "StudyInstanceUID", sids)
    return out


soft = pd.concat([load_soft("weak/claude_labels_300.csv", "weak/sample300.csv"),
                  load_soft("weak/claude_labels_remaining.csv", "weak/remaining.csv")],
                 ignore_index=True)
assert not set(soft["StudyInstanceUID"]) & set(clean["StudyInstanceUID"]), \
    "soft pool overlaps the 56 clean studies"
print(f"clean={len(clean)}  soft pool={len(soft)}", flush=True)


def pooled(sid):
    return np.load(f"{FEAT}/{sid}.npy").astype(np.float32).reshape(-1, 768).mean(0)


Xclean = torch.tensor(np.stack([pooled(x) for x in clean["StudyInstanceUID"]]))
Yclean = clean[TARGETS].to_numpy(np.float32)
Xsoft = torch.tensor(np.stack([pooled(x) for x in soft["StudyInstanceUID"]]))
Ysoft = torch.tensor(soft[TARGETS].to_numpy(np.float32))
print("features loaded", flush=True)


def macro_auc(yt, yp):
    aucs = []
    for i in range(len(TARGETS)):
        v = np.isfinite(yt[:, i]) & np.isfinite(yp[:, i])
        a, b = yt[v, i], yp[v, i]
        aucs.append(roc_auc_score(a, b) if len(np.unique(a)) == 2 else np.nan)
    fin = [x for x in aucs if np.isfinite(x)]
    return (float(np.mean(fin)) if fin else np.nan), aucs


def pos_weight(Y, cap=20.0):
    w = []
    for i in range(Y.shape[1]):
        v = Y[:, i].numpy()
        p, n = int((v == 1).sum()), int((v == 0).sum())
        w.append(1.0 if (p == 0 or n == 0) else min(n / p, cap))
    return torch.tensor(w, dtype=torch.float32)


def fit_predict(Xtr, Ytr, Wtr, Xev, Yev, seed):
    """Returns (final-epoch preds, best-epoch preds by fold AUC, best AUC)."""
    torch.manual_seed(seed)
    head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(),
                         nn.Dropout(0.3), nn.Linear(256, 12))
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    pw = pos_weight(Ytr)
    g = torch.Generator().manual_seed(seed + 1000)
    best, best_pred = -1.0, None
    for ep in range(EPOCHS):
        head.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), BS):
            idx = perm[i:i + BS]
            el = F.binary_cross_entropy_with_logits(
                head(Xtr[idx]), Ytr[idx], pos_weight=pw, reduction="none")
            loss = (el * Wtr[idx, None]).sum() / Wtr[idx].sum().clamp(min=1.0) / 12
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            p = torch.sigmoid(head(Xev)).numpy().astype(np.float64)
        a, _ = macro_auc(Yev.astype(np.float64), p)
        if np.isfinite(a) and a > best:
            best, best_pred = a, p
    return p, best_pred, best


def cv(n_soft, w_soft, seed, n_splits=5):
    strata = pd.qcut(pd.Series(Yclean.sum(1)).rank(method="first"),
                     q=n_splits, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20261020)
    oof_final = np.full_like(Yclean, np.nan, dtype=np.float64)
    oof_best = np.full_like(Yclean, np.nan, dtype=np.float64)
    for tr, ev in skf.split(np.zeros(len(clean)), strata):
        Xtr = torch.cat([Xclean[tr], Xsoft[:n_soft]])
        Ytr = torch.cat([torch.tensor(Yclean[tr]), Ysoft[:n_soft]])
        Wtr = torch.cat([torch.ones(len(tr)), torch.full((n_soft,), w_soft)])
        pf, pb, _ = fit_predict(Xtr, Ytr, Wtr, Xclean[ev], Yclean[ev], seed)
        oof_final[ev], oof_best[ev] = pf, pb
    yt = Yclean.astype(np.float64)
    return macro_auc(yt, oof_final)[0], macro_auc(yt, oof_best)[0], \
        macro_auc(yt, oof_final)[1]


if __name__ == "__main__":
    rows = []
    for n_soft in (0, len(Xsoft)):
        for w_soft in ([1.0] if n_soft == 0 else [1.0, 0.3]):
            fin, bst, per_t = [], [], []
            for seed in (0, 1, 2):
                f, b, pt = cv(n_soft, w_soft, seed)
                fin.append(f); bst.append(b); per_t.append(pt)
            row = dict(n_soft=n_soft, w_soft=w_soft,
                       primary_mean=float(np.mean(fin)), primary_sd=float(np.std(fin)),
                       primary=[round(x, 4) for x in fin],
                       secondary_mean=float(np.mean(bst)), secondary_sd=float(np.std(bst)),
                       secondary=[round(x, 4) for x in bst],
                       per_target_primary={
                           TARGETS[i]: (round(float(np.nanmean([p[i] for p in per_t])), 4)
                                        if np.isfinite([p[i] for p in per_t]).any() else None)
                           for i in range(12)})
            rows.append(row)
            print(f"n_soft={n_soft:5d} w={w_soft}: PRIMARY(56-study CV, fixed 80 ep) "
                  f"{row['primary_mean']:.4f} +- {row['primary_sd']:.4f} {row['primary']} | "
                  f"SECONDARY(best-epoch) {row['secondary_mean']:.4f} {row['secondary']}",
                  flush=True)
    json.dump(rows, open("weak/final_cv_results.json", "w"), indent=1)
    print("DONE")
