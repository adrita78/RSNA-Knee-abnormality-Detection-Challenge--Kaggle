"""Deliverable for the ADOPTED config (PLAN_08 Amendment 5, Lever B2):
3-plane concatenation (sagittal + coronal + axial), frozen ConvNeXt-Small,
mean pooling per plane, presence flags, trained on all 56 image-labeled
studies plus all 1,720 report-labeled ones at the fixed 80 epochs whose
generalisation was measured at 0.7627 +- 0.0036.

NOT a drop-in for the notebook's ConvNeXtFrozenPool: the head takes a 2,307-d
3-plane vector, not a 768-d single-plane one. Saved with the plane list and
feature recipe so inference is reproducible.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, json
import lever_b as B

torch.set_num_threads(1)
SEED, EPOCHS, BS, LR, WD = 0, 80, 16, 3e-4, 1e-4
PLANES = ["sagittal", "coronal", "axial"]
OUT = "checkpoints/head_3plane_weaklabels.pt"

Xc = B.build(B.clean["StudyInstanceUID"], PLANES)
Xw = B.build(B.weak["StudyInstanceUID"], PLANES)
X = torch.cat([Xc, Xw])
Y = torch.cat([torch.tensor(B.Yc), B.Yw])
print(f"train rows {X.shape[0]}, input dim {X.shape[1]}", flush=True)

pw = B.pos_weight(Y)
net = B.mlp(X.shape[1], SEED)
opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)
gen = torch.Generator().manual_seed(SEED + 1000)
for ep in range(EPOCHS):
    net.train()
    perm = torch.randperm(len(X), generator=gen)
    tot = 0.0
    for i in range(0, len(X), BS):
        j = perm[i:i+BS]
        el = F.binary_cross_entropy_with_logits(net(X[j]), Y[j], pos_weight=pw,
                                                reduction="none")
        loss = el.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.detach()) * len(j)
    if (ep + 1) % 20 == 0:
        print(f"epoch {ep+1:3d}  train loss {tot/len(X):.5f}", flush=True)

net.eval()
with torch.no_grad():
    p = torch.sigmoid(net(Xc)).numpy().astype(np.float64)
print(f"in-sample macro AUC on the 56 (NOT a performance estimate): "
      f"{B.macro_auc(B.Yc.astype(np.float64), p)[0]:.4f}")

torch.save({
    "epoch": EPOCHS,
    "head_state_dict": net.state_dict(),
    "input_dim": int(X.shape[1]),
    "planes": PLANES,
    "targets": B.TARGETS,
    "model_type": "Head3PlaneFrozenConvNeXt",
    "cv_macro_auc_56_studies": 0.7627,
    "cv_macro_auc_56_studies_sd": 0.0036,
    "feature_recipe": ("per plane: first fluid-sensitive fat-suppressed series "
                       "of that plane, every slice through frozen ImageNet "
                       "ConvNeXt-Small (classifier=Identity) -> 768-d, mean over "
                       "slices; concatenate planes in the order given, then "
                       "append one 0/1 presence flag per plane (zero vector when "
                       "a plane is absent). Final dim 3*768 + 3 = 2307."),
    "note": ("trained on 56 image-labeled + 1,720 report-labeled studies, "
             "fixed 80 epochs, no epoch selection. Quoted AUC is 5-fold CV over "
             "the 56 with out-of-fold predictions pooled (lever_b.py). Caveats: "
             "public JPEG-224 re-encode, not original DICOMs; report labels are "
             "model-read from text. Best of four architecture variants tried "
             "against the same 56 studies; information-free noise floor ~0.008."),
}, OUT)
print("saved", OUT)
ck = torch.load(OUT, map_location="cpu", weights_only=False)
chk = B.mlp(ck["input_dim"], 0); chk.load_state_dict(ck["head_state_dict"])
print("strict-load verified; dim", ck["input_dim"], "planes", ck["planes"])
