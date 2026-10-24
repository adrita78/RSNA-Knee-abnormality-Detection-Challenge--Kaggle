"""Cache frozen ConvNeXt-Small features for all 4,150 sagittal_fs series
(reads JPEGs from USB; ~768-dim per slice). Output: weak/feat/{study}.npy"""
import os, sys
import numpy as np, pandas as pd, torch
from PIL import Image
from torchvision import models, transforms
D = "/media/hyunsang/Ultra Touch/rsna-knee/kagglehub/datasets/alenic/rsna-knee-abnormality-detection-jpeg-224x224/versions/1"
OUT = "/home/hyunsang/Documents/adrita/10.19/local_run/weak/feat"
os.makedirs(OUT, exist_ok=True)
t = pd.read_csv(f"{D}/train.csv"); s = pd.read_csv(f"{D}/train_series.csv")
sfs = s[(s["Anatomical_Plane"].str.lower()=="sagittal") & (s["Fluid_Sensitive"]==1) & (s["Fat_Suppression"]==1)]
one = sfs.sort_values(["StudyInstanceUID","SeriesInstanceUID"]).drop_duplicates("StudyInstanceUID", keep="first")
model_df = t.merge(one[["StudyInstanceUID","SeriesInstanceUID"]], on="StudyInstanceUID", how="inner").reset_index(drop=True)
bb = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
bb.classifier = torch.nn.Identity(); bb.requires_grad_(False); bb.eval().cuda()
norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
n_done = 0
with torch.no_grad():
    for _, r in model_df.iterrows():
        su, se = r["StudyInstanceUID"], r["SeriesInstanceUID"]
        out = f"{OUT}/{su}.npy"
        if os.path.exists(out): n_done += 1; continue
        d = f"{D}/train_series/{su}/{se}"
        files = sorted(os.listdir(d))
        imgs = []
        for f in files:
            im = Image.open(os.path.join(d, f)).convert("RGB").resize((224,224))
            imgs.append(norm(transforms.functional.to_tensor(im)))
        x = torch.stack(imgs).cuda()
        feats = []
        for i in range(0, len(x), 32):
            feats.append(bb(x[i:i+32]).cpu())
        np.save(out, torch.cat(feats).numpy().astype(np.float16))
        n_done += 1
        if n_done % 200 == 0: print(f"{n_done}/{len(model_df)}", flush=True)
print(f"FEATURES DONE {n_done}")
