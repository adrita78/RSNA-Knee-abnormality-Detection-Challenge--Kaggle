"""Deliverable for the best SINGLE-PLANE config: axial + frozen ConvNeXt-Base.

CV 0.7611 +- 0.0090, which ties the 3-plane ConvNeXt-Small arm (0.7632) on one
plane instead of three. Trained on all 56 image-labelled + 1,720 report-labelled
studies, fixed 80 epochs, no epoch selection.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd, os
from sklearn.metrics import roc_auc_score
torch.set_num_threads(1)

D = ("/media/hyunsang/Ultra Touch/rsna-knee/kagglehub/datasets/alenic/"
     "rsna-knee-abnormality-detection-jpeg-224x224/versions/1")
FEAT, DIM = "weak/feat_axial_convnext_base", 1024
TARGETS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA",
           "Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]
SEED, EPOCHS, BS, LR, WD = 0, 80, 16, 3e-4, 1e-4
OUT = "checkpoints/head_axial_convnext_base_0761.pt"

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

def pooled(sid):
    return np.load(f"{FEAT}/{sid}.npy").astype(np.float32).reshape(-1, DIM).mean(0)

Xc = torch.tensor(np.stack([pooled(x) for x in clean["StudyInstanceUID"]]))
Xw = torch.tensor(np.stack([pooled(x) for x in weak["StudyInstanceUID"]]))
X = torch.cat([Xc, Xw])
Y = torch.cat([torch.tensor(clean[TARGETS].to_numpy(np.float32)),
               torch.tensor(weak[TARGETS].to_numpy(np.float32))])
print(f"train rows {len(X)}, dim {X.shape[1]}", flush=True)

ws=[]
for i in range(12):
    v=Y[:,i].numpy(); p,n=int((v==1).sum()),int((v==0).sum())
    ws.append(1.0 if (p==0 or n==0) else min(n/p,20.0))
pw = torch.tensor(ws, dtype=torch.float32)

torch.manual_seed(SEED)
head = nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM,256), nn.GELU(),
                     nn.Dropout(0.3), nn.Linear(256,12))
opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
g = torch.Generator().manual_seed(SEED+1000)
for ep in range(EPOCHS):
    head.train(); perm = torch.randperm(len(X), generator=g); tot=0.0
    for i in range(0,len(X),BS):
        j=perm[i:i+BS]
        el=F.binary_cross_entropy_with_logits(head(X[j]), Y[j], pos_weight=pw,
                                              reduction="none")
        loss=el.mean(); opt.zero_grad(); loss.backward(); opt.step()
        tot+=float(loss.detach())*len(j)
    if (ep+1)%20==0: print(f"epoch {ep+1:3d} loss {tot/len(X):.5f}", flush=True)

head.eval()
with torch.no_grad(): p=torch.sigmoid(head(Xc)).numpy().astype(np.float64)
yt=clean[TARGETS].to_numpy(np.float64)
a=[roc_auc_score(yt[:,i],p[:,i]) for i in range(12) if len(np.unique(yt[:,i]))==2]
print(f"in-sample macro AUC on the 56 (NOT a performance estimate): {np.mean(a):.4f}")

torch.save({"head_state_dict": head.state_dict(), "plane": "axial",
            "backbone": "convnext_base", "feature_dim": DIM, "targets": TARGETS,
            "arch": ("torchvision convnext_base(ConvNeXt_Base_Weights.DEFAULT), "
                     "classifier=Identity, mean-pool slice features, then this head"),
            "cv_macro_auc_56_studies": 0.7611, "cv_macro_auc_56_studies_sd": 0.0090,
            "note": ("BEST SINGLE-PLANE config measured: ties the 3-plane "
                     "ConvNeXt-Small arm (0.7632) using one plane instead of three. "
                     "The margin over ConvNeXt-Small axial (0.7514) is +0.010, which "
                     "is AT the measured noise floor, so treat it as a cheap "
                     "alternative rather than an established improvement. Feed it "
                     "the AXIAL fluid-sensitive fat-suppressed series.")},
           OUT)
print("saved", OUT)
