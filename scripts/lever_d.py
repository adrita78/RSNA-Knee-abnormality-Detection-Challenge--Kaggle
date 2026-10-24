"""LEVER D: Adrita's architecture, measured on the same footing.

Her CachedFeatureMultiPlaneBiLSTM (shared BiLSTM over cached per-slice
ConvNeXt features, per-plane learned attention pooling, learned embedding for
absent planes) copied from with-all-planes-weak-label.ipynb cell 38, evaluated
with the SAME protocol as every other arm in PLAN_08: 5-fold CV over all 56
image-labeled studies, out-of-fold pooled, 3 seeds, fixed 80 epochs, the 1,720
report labels included at weight 1.0.

Reference points (mean-pool head, same protocol):
    sagittal 0.7332   axial 0.7514   3-plane concat 0.7627
"""
import os, sys, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(1)
D = ("/media/hyunsang/Ultra Touch/rsna-knee/kagglehub/datasets/alenic/"
     "rsna-knee-abnormality-detection-jpeg-224x224/versions/1")
DIRS = {"sagittal": "weak/feat", "coronal": "weak/feat_coronal", "axial": "weak/feat_axial"}
TARGETS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA",
           "Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
EPOCHS, BS, LR, WD = 80, 16, 3e-4, 1e-4

ARM = sys.argv[1] if len(sys.argv) > 1 else "3plane"
PLANES = {"3plane": ["sagittal","coronal","axial"], "axial": ["axial"],
          "sagittal": ["sagittal"]}[ARM]
print(f"arm={ARM} planes={PLANES} device={DEV}", flush=True)

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
assert not set(weak["StudyInstanceUID"]) & set(clean["StudyInstanceUID"])

def slices(sid, plane):
    p = f"{DIRS[plane]}/{sid}.npy"
    if not os.path.exists(p): return None
    return torch.tensor(np.load(p).astype(np.float32).reshape(-1, 768))

print("loading per-slice features...", flush=True)
SID = list(clean["StudyInstanceUID"]) + list(weak["StudyInstanceUID"])
FEAT = {pl: {sid: slices(sid, pl) for sid in SID} for pl in PLANES}
for pl in PLANES:
    n = sum(1 for v in FEAT[pl].values() if v is not None)
    print(f"  {pl}: {n}/{len(SID)} present", flush=True)
Yc = clean[TARGETS].to_numpy(np.float32)
Yw = weak[TARGETS].to_numpy(np.float32)
Yall = np.concatenate([Yc, Yw])

class CachedFeatureMultiPlaneBiLSTM(nn.Module):
    """Verbatim from her notebook, cell 38; PLANES read from module scope."""
    def __init__(self, feature_dim=768, hidden_dim=128, num_layers=1,
                 num_targets=12, dropout=0.3, num_planes=3):
        super().__init__()
        self.num_planes = num_planes
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_dim,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.sequence_dim = 2*hidden_dim
        self.norm = nn.LayerNorm(self.sequence_dim)
        self.pool_queries = nn.Parameter(torch.randn(num_planes, self.sequence_dim)*0.02)
        self.missing_plane = nn.Parameter(torch.zeros(num_planes, self.sequence_dim))
        self.classifier = nn.Sequential(
            nn.Linear(num_planes*self.sequence_dim, 256), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(256, num_targets))

    def encode_plane(self, features, lengths, plane_present, plane_idx):
        B = features.shape[0]; device = features.device
        pooled = self.missing_plane[plane_idx].unsqueeze(0).expand(B, -1).clone()
        valid_idx = torch.where((lengths > 0) & plane_present.bool())[0]
        if valid_idx.numel() == 0: return pooled
        vf, vl = features[valid_idx], lengths[valid_idx].long()
        packed = pack_padded_sequence(vf, vl.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        seq, _ = pad_packed_sequence(out, batch_first=True)
        seq = self.norm(seq)
        scores = torch.einsum("bsd,d->bs", seq, self.pool_queries[plane_idx]) / (self.sequence_dim**0.5)
        pos = torch.arange(seq.shape[1], device=device).unsqueeze(0)
        scores = scores.masked_fill(~(pos < vl.to(device).unsqueeze(1)), float("-inf"))
        att = torch.softmax(scores, dim=1)
        pooled[valid_idx] = (seq * att.unsqueeze(-1)).sum(dim=1)
        return pooled

    def forward(self, features, lengths, plane_present):
        pf = [self.encode_plane(features[pl], lengths[pl], plane_present[pl], p)
              for p, pl in enumerate(PLANES)]
        return self.classifier(torch.cat(pf, dim=1))

def make_batch(sids):
    feats, lens, pres = {}, {}, {}
    for pl in PLANES:
        xs = [FEAT[pl][sid] for sid in sids]
        L = torch.tensor([0 if x is None else x.shape[0] for x in xs], dtype=torch.long)
        M = max(int(L.max().item()), 1)
        pad = torch.zeros(len(sids), M, 768)
        for i, x in enumerate(xs):
            if x is not None: pad[i, :x.shape[0]] = x
        # HER NOTEBOOK'S BUG IS HERE, not in the model: prepare_multiplane_inputs
        # leaves lengths on CPU while plane_present goes to the GPU, so
        # encode_plane's `(lengths > 0) & plane_present.bool()` dies. The model
        # already calls vl.cpu() where pack_padded_sequence needs CPU lengths,
        # so lengths should arrive on DEVICE.
        feats[pl] = pad.to(DEV); lens[pl] = L.to(DEV)
        pres[pl] = torch.tensor([x is not None for x in xs]).to(DEV)
    return feats, lens, pres

def macro_auc(yt, yp):
    a=[]
    for i in range(12):
        ok = np.isfinite(yt[:,i]) & np.isfinite(yp[:,i]); y,p = yt[ok,i], yp[ok,i]
        a.append(roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan)
    fin=[x for x in a if np.isfinite(x)]
    return (float(np.mean(fin)) if fin else np.nan), a

def pos_weight(Y, cap=20.0):
    w=[]
    for i in range(Y.shape[1]):
        p,n=int((Y[:,i]==1).sum()),int((Y[:,i]==0).sum())
        w.append(1.0 if (p==0 or n==0) else min(n/p,cap))
    return torch.tensor(w, dtype=torch.float32, device=DEV)

def cv(seed, n_splits=5):
    strata = pd.qcut(pd.Series(Yc.sum(1)).rank(method="first"), q=n_splits, labels=False)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=20261020)
    oof = np.full_like(Yc, np.nan, dtype=np.float64)
    for tr, ev in skf.split(np.zeros(len(Yc)), strata):
        tr_sids = [clean["StudyInstanceUID"][i] for i in tr] + list(weak["StudyInstanceUID"])
        tr_Y = torch.tensor(np.concatenate([Yc[tr], Yw])).to(DEV)
        ev_sids = [clean["StudyInstanceUID"][i] for i in ev]
        pw = pos_weight(np.concatenate([Yc[tr], Yw]))
        torch.manual_seed(seed)
        net = CachedFeatureMultiPlaneBiLSTM(num_planes=len(PLANES)).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
        g = torch.Generator().manual_seed(seed+1000)
        for ep in range(EPOCHS):
            net.train()
            perm = torch.randperm(len(tr_sids), generator=g).tolist()
            for i in range(0, len(perm), BS):
                j = perm[i:i+BS]
                f, l, pr = make_batch([tr_sids[k] for k in j])
                el = F.binary_cross_entropy_with_logits(
                    net(f,l,pr), tr_Y[j], pos_weight=pw, reduction="none")
                loss = el.mean()
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            f,l,pr = make_batch(ev_sids)
            oof[ev] = torch.sigmoid(net(f,l,pr)).float().cpu().numpy().astype(np.float64)
    return macro_auc(Yc.astype(np.float64), oof)

if __name__ == "__main__":
    res = [cv(s) for s in (0,1,2)]
    a = [r[0] for r in res]
    per = np.nanmean([r[1] for r in res], axis=0)
    out = dict(arm=ARM, planes=PLANES, mean=float(np.mean(a)), sd=float(np.std(a)),
               aucs=[round(x,4) for x in a],
               per_target={TARGETS[i]: round(float(per[i]),4) for i in range(12)})
    json.dump(out, open(f"weak/lever_d_{ARM}.json","w"), indent=1)
    print(f"\nHER ARCHITECTURE ({ARM}): {np.mean(a):.4f} +- {np.std(a):.4f}  {out['aucs']}")
    print("mean-pool head refs: sagittal 0.7332 | axial 0.7514 | 3-plane 0.7627")
    print("DONE")
