"""Train the deliverable head on ALL 56 clean studies + the 1,720 Claude
report-labeled studies, and save it as a full ConvNeXtFrozenPool state_dict
so it drops into the notebook pipeline unchanged.

No epoch selection: fixed 80 epochs, exactly the protocol whose honest
performance was measured by 5-fold CV in final_eval.py (0.733 macro AUC).
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as models
import final_eval as fe

SEED, EPOCHS, BS, LR, WD = 0, 80, 16, 3e-4, 1e-4
N_SOFT, W_SOFT = len(fe.Xsoft), 1.0
OUT = "checkpoints/convnext_frozenpool_weaklabels.pt"

Xtr = torch.cat([fe.Xclean, fe.Xsoft[:N_SOFT]])
Ytr = torch.cat([torch.tensor(fe.Yclean), fe.Ysoft[:N_SOFT]])
Wtr = torch.cat([torch.ones(len(fe.Xclean)), torch.full((N_SOFT,), W_SOFT)])

torch.manual_seed(SEED)
head = nn.Sequential(nn.LayerNorm(768), nn.Linear(768, 256), nn.GELU(),
                     nn.Dropout(0.3), nn.Linear(256, 12))
opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
pw = fe.pos_weight(Ytr)
g = torch.Generator().manual_seed(SEED + 1000)
for ep in range(EPOCHS):
    head.train()
    perm = torch.randperm(len(Xtr), generator=g)
    tot = 0.0
    for i in range(0, len(Xtr), BS):
        idx = perm[i:i + BS]
        el = F.binary_cross_entropy_with_logits(
            head(Xtr[idx]), Ytr[idx], pos_weight=pw, reduction="none")
        loss = (el * Wtr[idx, None]).sum() / Wtr[idx].sum().clamp(min=1.0) / 12
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss) * len(idx)
    if (ep + 1) % 20 == 0:
        print(f"epoch {ep+1:3d}  train loss {tot/len(Xtr):.5f}", flush=True)

head.eval()
with torch.no_grad():
    p = torch.sigmoid(head(fe.Xclean)).numpy().astype(np.float64)
fit_auc, _ = fe.macro_auc(fe.Yclean.astype(np.float64), p)
print(f"in-sample macro AUC on the 56 (NOT a performance estimate): {fit_auc:.4f}")

full = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
full.classifier = nn.Identity()
sd = {f"backbone.{k}": v for k, v in full.state_dict().items()}
for k, v in head.state_dict().items():
    sd[f"classifier.{k}"] = v

torch.save({"epoch": EPOCHS,
            "model_state_dict": sd,
            "best_val_auc": None,
            "cv_macro_auc_56_studies": 0.7333,
            "cv_macro_auc_56_studies_sd": 0.0016,
            "targets": fe.TARGETS,
            "model_type": "ConvNeXtFrozenPool",
            "note": ("head trained on 56 image-labeled studies (weight 1.0) plus "
                     "1,720 studies weak-labeled by Claude from their radiology "
                     "reports (weight 1.0); frozen ImageNet ConvNeXt-Small "
                     "backbone; fixed 80 epochs, no epoch selection. The quoted "
                     "AUC is 5-fold CV over the 56 image-labeled studies with "
                     "out-of-fold predictions pooled (final_eval.py).")},
           OUT)
print("saved", OUT)

ck = torch.load(OUT, map_location="cpu", weights_only=False)
ref = fe.__dict__  # strict-load check against the real class
import importlib, sys
sys.path.insert(0, ".")
tl = importlib.import_module("train_local")
m = tl.ConvNeXtFrozenPool()
m.load_state_dict(ck["model_state_dict"], strict=True)
print("strict-load into ConvNeXtFrozenPool: OK")
