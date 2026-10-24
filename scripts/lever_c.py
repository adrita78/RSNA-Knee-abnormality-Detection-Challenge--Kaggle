"""LEVER C (PLAN_08 Amendment 5): unfreeze the backbone.

Architecture is identical to the frozen pipeline -- per-slice ConvNeXt
feature, mean over slices, same head -- so the only change is which weights
receive gradients. Sagittal_fs only; reference is 0.7332 (frozen, 1 plane).

Resumable: each (seed, fold) writes its out-of-fold predictions to
weak/lever_c_oof/, so a crash or a kill costs at most one fold.
"""
import os, sys, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

D = ("/media/hyunsang/Ultra Touch/rsna-knee/kagglehub/datasets/alenic/"
     "rsna-knee-abnormality-detection-jpeg-224x224/versions/1")
OOF = "weak/lever_c_oof"; os.makedirs(OOF, exist_ok=True)
TARGETS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA",
           "Lateral OA","PF OA","Effusion","Synovitis","Baker's",
           "Contusion","Fracture"]
EPOCHS, MICRO, ACCUM = 10, 1, 16        # effective batch 16 studies
CHUNK = 24   # slices per checkpointed backbone call. Slice counts reach 320,
# so a whole study in one graph OOMs; checkpointing recomputes activations in
# backward and bounds peak memory by CHUNK instead of by the study's length.
# All slices are still used, so the input matches the frozen arm exactly.
LR_BB, LR_HEAD, WD = 1e-5, 3e-4, 1e-4
DEV = "cuda"

t = pd.read_csv(f"{D}/train.csv"); s = pd.read_csv(f"{D}/train_series.csv")
sfs = s[(s["Anatomical_Plane"].str.lower()=="sagittal")
        & (s["Fluid_Sensitive"]==1) & (s["Fat_Suppression"]==1)]
one = (sfs.sort_values(["StudyInstanceUID","SeriesInstanceUID"])
          .drop_duplicates("StudyInstanceUID", keep="first"))
model_df = t.merge(one[["StudyInstanceUID","SeriesInstanceUID"]],
                   on="StudyInstanceUID", how="inner").reset_index(drop=True)
clean = model_df.loc[model_df[TARGETS].notna().all(axis=1)].reset_index(drop=True)
weak = pd.read_csv("weak/weak_labels_by_study.csv").merge(
    model_df[["StudyInstanceUID","SeriesInstanceUID"]], on="StudyInstanceUID",
    how="inner").reset_index(drop=True)
assert len(clean)==56 and len(weak)==1720
assert not set(weak["StudyInstanceUID"]) & set(clean["StudyInstanceUID"])

Yc = clean[TARGETS].to_numpy(np.float32)
Yw = weak[TARGETS].to_numpy(np.float32)

tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485,0.456,0.406],
                                              std=[0.229,0.224,0.225])])

class Studies(Dataset):
    def __init__(self, df, Y):
        self.df = df.reset_index(drop=True); self.Y = Y
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        d = f"{D}/train_series/{r['StudyInstanceUID']}/{r['SeriesInstanceUID']}"
        xs = [tf(Image.open(os.path.join(d,f)).convert("RGB"))
              for f in sorted(os.listdir(d))]
        return torch.stack(xs), torch.tensor(self.Y[i])

def collate(b):
    xs = torch.cat([x for x,_ in b])                 # (sum_S,3,224,224)
    lens = torch.tensor([x.shape[0] for x,_ in b])
    ys = torch.stack([y for _,y in b])
    return xs, lens, ys

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
        bb.classifier = nn.Identity()
        self.backbone = bb
        self.head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768,256),
                                  nn.GELU(), nn.Dropout(0.3), nn.Linear(256,12))
    def _feats(self, xs):
        if not self.training:
            return self.backbone(xs).reshape(xs.shape[0], -1)
        outs = []
        for i in range(0, xs.shape[0], CHUNK):
            c = xs[i:i+CHUNK]
            outs.append(torch.utils.checkpoint.checkpoint(
                lambda z: self.backbone(z).reshape(z.shape[0], -1),
                c, use_reentrant=False))
        return torch.cat(outs)

    def forward(self, xs, lens):
        f = self._feats(xs)                                  # (sum_S,768)
        out, i = [], 0
        for L in lens.tolist():
            out.append(f[i:i+L].mean(0)); i += L
        return self.head(torch.stack(out))

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
        p,n = int((Y[:,i]==1).sum()), int((Y[:,i]==0).sum())
        w.append(1.0 if (p==0 or n==0) else min(n/p,cap))
    return torch.tensor(w, dtype=torch.float32, device=DEV)

def run(seed, fold, tr_idx, ev_idx, smoke=False):
    tag = f"{OOF}/s{seed}_f{fold}.npy"
    if os.path.exists(tag) and not smoke:
        print(f"  seed{seed} fold{fold}: cached", flush=True); return np.load(tag)
    tr_df = pd.concat([clean.iloc[tr_idx][["StudyInstanceUID","SeriesInstanceUID"]],
                       weak[["StudyInstanceUID","SeriesInstanceUID"]]], ignore_index=True)
    tr_Y = np.concatenate([Yc[tr_idx], Yw])
    if smoke:
        tr_df, tr_Y = tr_df.iloc[:40].reset_index(drop=True), tr_Y[:40]
    pw = pos_weight(tr_Y)
    torch.manual_seed(seed)
    net = Net().to(DEV)
    opt = torch.optim.AdamW([{"params": net.backbone.parameters(), "lr": LR_BB},
                             {"params": net.head.parameters(), "lr": LR_HEAD}],
                            weight_decay=WD)
    scaler = torch.amp.GradScaler("cuda")
    dl = DataLoader(Studies(tr_df, tr_Y), batch_size=MICRO, shuffle=True,
                    num_workers=4, collate_fn=collate, drop_last=False,
                    generator=torch.Generator().manual_seed(seed+1000))
    n_ep = 1 if smoke else EPOCHS
    for ep in range(n_ep):
        net.train(); t0=time.time(); tot=0.0; nb=0
        opt.zero_grad(set_to_none=True)
        for k,(xs,lens,ys) in enumerate(dl):
            xs, ys = xs.to(DEV, non_blocking=True), ys.to(DEV)
            with torch.amp.autocast("cuda"):
                el = F.binary_cross_entropy_with_logits(
                    net(xs,lens), ys, pos_weight=pw, reduction="none")
                loss = el.mean() / ACCUM
            scaler.scale(loss).backward()
            if (k+1) % ACCUM == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            tot += float(loss.detach())*ACCUM; nb += 1
            if smoke and k >= 9: break
        print(f"  seed{seed} fold{fold} ep{ep+1}/{n_ep} loss {tot/max(nb,1):.4f} "
              f"{(time.time()-t0)/60:.1f} min  mem {torch.cuda.max_memory_allocated()/1e9:.1f}GB",
              flush=True)
    net.eval()
    ev_df = clean.iloc[ev_idx][["StudyInstanceUID","SeriesInstanceUID"]]
    dle = DataLoader(Studies(ev_df, Yc[ev_idx]), batch_size=MICRO, shuffle=False,
                     num_workers=4, collate_fn=collate)
    preds=[]
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for xs,lens,_ in dle:
            preds.append(torch.sigmoid(net(xs.to(DEV),lens)).float().cpu().numpy())
    p = np.concatenate(preds).astype(np.float64)
    if not smoke: np.save(tag, p)
    del net, opt; torch.cuda.empty_cache()
    return p

if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    strata = pd.qcut(pd.Series(Yc.sum(1)).rank(method="first"), q=5, labels=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20261020)
    folds = list(skf.split(np.zeros(len(Yc)), strata))
    if smoke:
        tr, ev = folds[0]
        p = run(0, 0, tr, ev, smoke=True)
        print(f"SMOKE OK preds {p.shape} range [{p.min():.3f},{p.max():.3f}]")
        sys.exit(0)
    rows=[]
    for seed in (0,1,2):
        oof = np.full_like(Yc, np.nan, dtype=np.float64)
        for fi,(tr,ev) in enumerate(folds):
            oof[ev] = run(seed, fi, tr, ev)
        auc, per = macro_auc(Yc.astype(np.float64), oof)
        rows.append(dict(seed=seed, auc=round(auc,4),
                         per_target={TARGETS[i]: round(float(per[i]),4) for i in range(12)}))
        print(f"SEED {seed}: {auc:.4f}", flush=True)
        json.dump(rows, open("weak/lever_c_results.json","w"), indent=1)
    a=[r["auc"] for r in rows]
    print(f"LEVER C: {np.mean(a):.4f} +- {np.std(a):.4f}  {a}")
    print("reference (frozen, 1 plane): 0.7332 +- 0.0016")
    print("DONE")
