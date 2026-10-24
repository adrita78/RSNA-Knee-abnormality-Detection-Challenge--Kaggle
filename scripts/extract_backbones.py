"""Is the FROZEN FEATURE QUALITY the ceiling?

Adrita trains with a frozen backbone, as do we. Every lever that rearranged
fixed features failed; the only thing that worked added information. So: swap
the frozen backbone and see whether the number moves at all.

Design: a WEAKER backbone (ResNet-50) and a STRONGER one (ConvNeXt-Base)
either side of the current ConvNeXt-Small. If there is a clear monotone trend,
feature quality matters and fine-tuning is worth its 21 h. If all three tie,
feature quality is NOT the bottleneck and fine-tuning probably will not help
either -- which settles Lever C for good, cheaply.

Axial plane only (best single plane, 100% study coverage).
"""
import os, sys
import numpy as np, pandas as pd, torch
from PIL import Image
from torchvision import models, transforms

D = ("/media/hyunsang/Ultra Touch/rsna-knee/kagglehub/datasets/alenic/"
     "rsna-knee-abnormality-detection-jpeg-224x224/versions/1")
BASE = "/home/hyunsang/Documents/adrita/10.19/local_run"
TARGETS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA",
           "Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]

SPECS = {
 "resnet50":      (lambda: models.resnet50(weights=models.ResNet50_Weights.DEFAULT), 2048),
 "convnext_base": (lambda: models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT), 1024),
}

t = pd.read_csv(f"{D}/train.csv"); s = pd.read_csv(f"{D}/train_series.csv")
clean = set(t.loc[t[TARGETS].notna().all(axis=1), "StudyInstanceUID"])
weak = set(pd.read_csv(f"{BASE}/weak/weak_labels_by_study.csv")["StudyInstanceUID"])
need = clean | weak
sel = s[(s["Anatomical_Plane"].str.lower()=="axial")
        & (s["Fluid_Sensitive"]==1) & (s["Fat_Suppression"]==1)]
one = (sel.sort_values(["StudyInstanceUID","SeriesInstanceUID"])
          .drop_duplicates("StudyInstanceUID", keep="first"))
one = one[one["StudyInstanceUID"].isin(need)].reset_index(drop=True)
print(f"axial series to featurize: {len(one)}", flush=True)

norm = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

for name in (sys.argv[1:] or list(SPECS)):
    ctor, dim = SPECS[name]
    out_dir = f"{BASE}/weak/feat_axial_{name}"
    os.makedirs(out_dir, exist_ok=True)
    bb = ctor()
    if name.startswith("convnext"): bb.classifier = torch.nn.Identity()
    else: bb.fc = torch.nn.Identity()
    bb.requires_grad_(False); bb.eval().cuda()
    done = fail = 0
    with torch.no_grad():
        for _, r in one.iterrows():
            dst = f"{out_dir}/{r['StudyInstanceUID']}.npy"
            if os.path.exists(dst): done += 1; continue
            d = f"{D}/train_series/{r['StudyInstanceUID']}/{r['SeriesInstanceUID']}"
            try:
                imgs = [norm(transforms.functional.to_tensor(
                            Image.open(os.path.join(d,f)).convert("RGB").resize((224,224))))
                        for f in sorted(os.listdir(d))]
                x = torch.stack(imgs).cuda()
                f = torch.cat([bb(x[i:i+32]).reshape(min(32, len(x)-i), -1).cpu()
                               for i in range(0, len(x), 32)])
                assert f.shape[1] == dim, f"{name}: got {f.shape[1]} expected {dim}"
                np.save(dst, f.numpy().astype(np.float16)); done += 1
            except Exception as e:
                fail += 1
                if fail <= 3: print(f"  fail {r['StudyInstanceUID']}: {e}", flush=True)
            if done % 400 == 0: print(f"  {name} {done}/{len(one)}", flush=True)
    print(f"{name} DONE: {done} cached ({dim}-d), {fail} failed", flush=True)
    del bb; torch.cuda.empty_cache()
print("ALL BACKBONES DONE")
