"""Deliverable for the CORRECTED recommendation: axial_fs single plane.

Lever B's diagnostic showed axial alone (0.7623 +- 0.0051) ties the 3-plane
concatenation (0.7627 +- 0.0036) while costing exactly what the original
single-plane pipeline costs. So the actionable change is the PLANE, not
multi-plane fusion.

Same architecture as the notebook's ConvNeXtFrozenPool, so this saves a full
drop-in state_dict. The ONLY difference from the shipped 0.733 checkpoint is
which series each study contributes: axial + Fluid_Sensitive + Fat_Suppression
instead of sagittal.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import models
import lever_b as B

torch.set_num_threads(1)
SEED, EPOCHS, BS, LR, WD = 0, 80, 16, 3e-4, 1e-4
OUT = "checkpoints/convnext_frozenpool_axial_weaklabels.pt"

Xc = B.build(B.clean["StudyInstanceUID"], ["axial"])      # dim 769 (+1 flag)
Xw = B.build(B.weak["StudyInstanceUID"], ["axial"])
# drop the presence flag: all studies have axial, so it is a constant column,
# and dropping it keeps the head a drop-in 768-d ConvNeXtFrozenPool classifier
Xc, Xw = Xc[:, :768].contiguous(), Xw[:, :768].contiguous()
X = torch.cat([Xc, Xw]); Y = torch.cat([torch.tensor(B.Yc), B.Yw])
print(f"train rows {X.shape[0]}, dim {X.shape[1]}", flush=True)

pw = B.pos_weight(Y)
head = B.mlp(768, SEED)
opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
gen = torch.Generator().manual_seed(SEED + 1000)
for ep in range(EPOCHS):
    head.train()
    perm = torch.randperm(len(X), generator=gen); tot = 0.0
    for i in range(0, len(X), BS):
        j = perm[i:i+BS]
        el = F.binary_cross_entropy_with_logits(head(X[j]), Y[j], pos_weight=pw,
                                                reduction="none")
        loss = el.mean()
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.detach())*len(j)
    if (ep+1) % 20 == 0:
        print(f"epoch {ep+1:3d}  train loss {tot/len(X):.5f}", flush=True)

head.eval()
with torch.no_grad():
    p = torch.sigmoid(head(Xc)).numpy().astype(np.float64)
print(f"in-sample macro AUC on the 56 (NOT a performance estimate): "
      f"{B.macro_auc(B.Yc.astype(np.float64), p)[0]:.4f}")

full = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
full.classifier = nn.Identity()
sd = {f"backbone.{k}": v for k, v in full.state_dict().items()}
for k, v in head.state_dict().items():
    sd[f"classifier.{k}"] = v
torch.save({"epoch": EPOCHS, "model_state_dict": sd, "targets": B.TARGETS,
            "model_type": "ConvNeXtFrozenPool",
            "series_selection": ("Anatomical_Plane == 'axial' AND "
                                 "Fluid_Sensitive == 1 AND Fat_Suppression == 1, "
                                 "first series per study"),
            "cv_macro_auc_56_studies": 0.7623,
            "cv_macro_auc_56_studies_sd": 0.0051,
            "note": ("IMPORTANT: feed this model the AXIAL fluid-sensitive "
                     "fat-suppressed series, not the sagittal one. Architecture "
                     "is identical to the sagittal 0.733 checkpoint; only the "
                     "input plane differs. Head trained on 56 image-labeled + "
                     "1,720 report-labeled studies, fixed 80 epochs, no epoch "
                     "selection. Public JPEG-224 re-encode, not DICOMs.")},
           OUT)
print("saved", OUT)
ck = torch.load(OUT, map_location="cpu", weights_only=False)
import importlib, sys; sys.path.insert(0, ".")
tl = importlib.import_module("train_local")
m = tl.ConvNeXtFrozenPool(); m.load_state_dict(ck["model_state_dict"], strict=True)
print("strict-load into ConvNeXtFrozenPool: OK")
