"""LEVER B prep: cache frozen ConvNeXt features for the coronal_fs and
axial_fs series of the 1,776 studies we train on (56 clean + 1,720 weak).
Mirrors extract_features.py exactly, one output dir per plane."""
import os, sys
import numpy as np, pandas as pd, torch
from PIL import Image
from torchvision import models, transforms

D = ("/media/hyunsang/Ultra Touch/rsna-knee/kagglehub/datasets/alenic/"
     "rsna-knee-abnormality-detection-jpeg-224x224/versions/1")
BASE = "/home/hyunsang/Documents/adrita/10.19/local_run"
TARGETS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA",
           "Lateral OA","PF OA","Effusion","Synovitis","Baker's",
           "Contusion","Fracture"]

t = pd.read_csv(f"{D}/train.csv"); s = pd.read_csv(f"{D}/train_series.csv")
clean = set(t.loc[t[TARGETS].notna().all(axis=1), "StudyInstanceUID"])
weak = set(pd.read_csv(f"{BASE}/weak/weak_labels_by_study.csv")["StudyInstanceUID"])
need = clean | weak
print(f"studies needed: {len(need)}", flush=True)

bb = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
bb.classifier = torch.nn.Identity(); bb.requires_grad_(False); bb.eval().cuda()
norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

for plane in ("coronal", "axial"):
    out_dir = f"{BASE}/weak/feat_{plane}"
    os.makedirs(out_dir, exist_ok=True)
    sel = s[(s["Anatomical_Plane"].str.lower() == plane)
            & (s["Fluid_Sensitive"] == 1) & (s["Fat_Suppression"] == 1)]
    one = (sel.sort_values(["StudyInstanceUID","SeriesInstanceUID"])
              .drop_duplicates("StudyInstanceUID", keep="first"))
    one = one[one["StudyInstanceUID"].isin(need)].reset_index(drop=True)
    print(f"\n{plane}: {len(one)} of {len(need)} studies have such a series", flush=True)

    done = fail = 0
    with torch.no_grad():
        for _, r in one.iterrows():
            su, se = r["StudyInstanceUID"], r["SeriesInstanceUID"]
            dst = f"{out_dir}/{su}.npy"
            if os.path.exists(dst):
                done += 1; continue
            d = f"{D}/train_series/{su}/{se}"
            try:
                files = sorted(os.listdir(d))
                imgs = [norm(transforms.functional.to_tensor(
                            Image.open(os.path.join(d,f)).convert("RGB").resize((224,224))))
                        for f in files]
                x = torch.stack(imgs).cuda()
                feats = [bb(x[i:i+32]).cpu() for i in range(0,len(x),32)]
                np.save(dst, torch.cat(feats).numpy().astype(np.float16))
                done += 1
            except Exception as e:
                fail += 1
                if fail <= 5: print(f"  fail {su}: {e}", flush=True)
            if done % 200 == 0:
                print(f"  {plane} {done}/{len(one)}", flush=True)
    print(f"{plane} DONE: {done} cached, {fail} failed", flush=True)
print("\nALL PLANES DONE")
