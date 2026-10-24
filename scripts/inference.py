"""Inference / submission generator.

Loads one of the small head checkpoints in checkpoints/, rebuilds the frozen
ConvNeXt-Small backbone from torchvision (the backbone in every checkpoint is
bit-identical to the stock ImageNet weights, verified, which is why only the
0.8 MB head is shipped), and writes a submission.csv.

    python inference.py --head checkpoints/head_axial_0751.pt \
                        --data /kaggle/input/competitions/rsna-knee-abnormality-detection \
                        --out submission.csv

PLANE COVERAGE MATTERS AT INFERENCE. On the training set, axial
fluid-sensitive+fat-suppressed exists for 4,407/4,407 studies but sagittal
for only 4,150/4,407 -- 257 studies have NO sagittal series. A sagittal model
therefore has no input for ~6% of studies. --fallback (default on) uses the
next available plane for those; without it they get 0.5 and contribute
nothing but noise. The head was trained on one plane, so a fallback plane is
a degraded guess, not an equivalent input -- which is the argument for using
the axial head in the first place.
"""
import argparse, os, sys
import numpy as np, pandas as pd, torch, torch.nn as nn
from torchvision import models, transforms
from PIL import Image

TARGETS = ["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA",
           "Lateral OA","PF OA","Effusion","Synovitis","Baker's",
           "Contusion","Fracture"]
TF = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                         transforms.Normalize(mean=[0.485,0.456,0.406],
                                              std=[0.229,0.224,0.225])])

BACKBONES = {
    "convnext_small": (lambda: models.convnext_small(
        weights=models.ConvNeXt_Small_Weights.DEFAULT), 768),
    "convnext_base": (lambda: models.convnext_base(
        weights=models.ConvNeXt_Base_Weights.DEFAULT), 1024),
    "resnet50": (lambda: models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT), 2048),
}


def build(head_path, device):
    ck = torch.load(head_path, map_location="cpu", weights_only=False)
    name = ck.get("backbone", "convnext_small")
    ctor, dim = BACKBONES[name]
    bb = ctor()
    if name.startswith("convnext"):
        bb.classifier = nn.Identity()
    else:
        bb.fc = nn.Identity()
    assert ck.get("feature_dim", dim) == dim, "head/backbone dimension mismatch"
    head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 256), nn.GELU(),
                         nn.Dropout(0.3), nn.Linear(256, 12))
    head.load_state_dict(ck["head_state_dict"])
    return bb.eval().to(device), head.eval().to(device), ck

def read_slices(d):
    """DICOM dir (competition) or JPEG dir (public re-encode)."""
    files = sorted(os.listdir(d))
    if files and files[0].lower().endswith((".jpg",".jpeg",".png")):
        return [TF(Image.open(os.path.join(d,f)).convert("RGB")) for f in files]
    import pydicom
    ds = []
    for f in files:
        p = os.path.join(d,f)
        if not os.path.isfile(p): continue
        try: x = pydicom.dcmread(p)
        except Exception: continue
        if hasattr(x,"PixelData"): ds.append(x)
    if not ds: raise RuntimeError(f"no readable slices in {d}")
    if hasattr(ds[0],"ImageOrientationPatient") and hasattr(ds[0],"ImagePositionPatient"):
        iop = np.asarray(ds[0].ImageOrientationPatient, dtype=np.float32)
        nrm = np.cross(iop[:3], iop[3:])
        ds = [ds[i] for i in np.argsort([float(np.dot(
            np.asarray(x.ImagePositionPatient,dtype=np.float32), nrm)) for x in ds])]
    elif all(hasattr(x,"InstanceNumber") for x in ds):
        ds = sorted(ds, key=lambda x: int(x.InstanceNumber))
    out = []
    for x in ds:
        a = x.pixel_array.astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = (a-lo)/(hi-lo) if hi > lo else np.zeros_like(a)
        out.append(TF(Image.fromarray((a*255).astype(np.uint8)).convert("RGB")))
    return out

def pick_series(series_df, plane, fallback):
    """One series per study for `plane`, then fallback planes for the rest."""
    order = [plane] + ([p for p in ("axial","coronal","sagittal") if p != plane]
                       if fallback else [])
    chosen, used = {}, {}
    for pl in order:
        sel = series_df[(series_df["Anatomical_Plane"].str.lower() == pl)
                        & (series_df["Fluid_Sensitive"] == 1)
                        & (series_df["Fat_Suppression"] == 1)]
        one = (sel.sort_values(["StudyInstanceUID","SeriesInstanceUID"])
                  .drop_duplicates("StudyInstanceUID", keep="first"))
        for _, r in one.iterrows():
            sid = r["StudyInstanceUID"]
            if sid not in chosen:
                chosen[sid] = r["SeriesInstanceUID"]; used[sid] = pl
    return chosen, used

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--split", default="test", choices=["test","train"])
    ap.add_argument("--no-fallback", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bb, head, ck = build(a.head, dev)
    plane = ck["plane"]
    print(f"head {os.path.basename(a.head)}  plane={plane}  "
          f"backbone={ck.get('backbone','convnext_small')}  "
          f"cv={ck.get('cv_macro_auc_56_studies')}  device={dev}")

    studies = pd.read_csv(f"{a.data}/{a.split}.csv")["StudyInstanceUID"].tolist()
    series = pd.read_csv(f"{a.data}/{a.split}_series.csv")
    chosen, used = pick_series(series, plane, not a.no_fallback)
    n_native = sum(1 for s in studies if used.get(s) == plane)
    print(f"{len(studies)} studies: {n_native} on {plane}, "
          f"{sum(1 for s in studies if s in chosen)-n_native} on a fallback plane, "
          f"{sum(1 for s in studies if s not in chosen)} with no usable series")

    rows = []
    for i, sid in enumerate(studies):
        if sid not in chosen:
            rows.append([sid] + [0.5]*12); continue
        d = f"{a.data}/{a.split}_series/{sid}/{chosen[sid]}"
        try:
            xs = torch.stack(read_slices(d)).to(dev)
        except Exception as e:
            print(f"  {sid}: {e} -> 0.5"); rows.append([sid]+[0.5]*12); continue
        with torch.no_grad():
            f = torch.cat([bb(xs[j:j+a.batch]).reshape(min(a.batch, len(xs)-j), -1)
                           for j in range(0, len(xs), a.batch)])  # (S, feature_dim)
            p = torch.sigmoid(head(f.mean(0, keepdim=True)))[0].float().cpu().numpy()
        rows.append([sid] + [float(v) for v in p])
        if (i+1) % 200 == 0: print(f"  {i+1}/{len(studies)}", flush=True)

    sub = pd.DataFrame(rows, columns=["StudyInstanceUID"] + TARGETS)
    sub.to_csv(a.out, index=False)
    print(f"wrote {a.out}  shape {sub.shape}")
    print(sub.head(3).to_string(index=False))

if __name__ == "__main__":
    main()
