# Running the two notebooks on Kaggle

Both need the competition data attached, plus the weak labels as a dataset.
You already have those uploaded at
`/kaggle/input/datasets/addy789/weak-labels-by-study/weak_labels_by_study.csv`;
`../labels/weak_labels_by_study.csv` here is the same file.

## single-plane-weak-labels.ipynb — the 0.733 pipeline

Run all. Section 4 reads DICOMs for 1,776 studies and takes 30-60 min, caching
as it goes, so a timed-out session can simply be re-run. Everything after that
is seconds, because the backbone is frozen and features are computed once.
Writes a full ConvNeXtFrozenPool state dict at the end.

Section 9 is the part worth reading: why the headline is 0.733 and not the
0.80 that an 11-12 study split reports.

## with-all-planes-weak-label-FIXED.ipynb — your notebook, unblocked

Defaults to `PLANES = ["axial_fs"]`, which is 1,778 series. The three-plane
line is commented directly beneath it if you want your original version; that
costs about 3x the extraction.

The five fixes are marked **FIX** in the cells that contain them, and listed
in the first markdown cell. Your model class, label-merge logic, Dataset and
collate are unchanged.

## Inference without retraining

`../scripts/inference.py` loads one of the 0.8 MB heads, rebuilds the frozen
backbone from torchvision and writes a submission.csv:

    python inference.py --head ../checkpoints/head_axial_convnext_base_0761.pt \
        --data /kaggle/input/competitions/rsna-knee-abnormality-detection \
        --out submission.csv

Each head records its own backbone and inference.py builds it, so swapping
head_axial_0751.pt (ConvNeXt-Small) in there works unchanged. The 3-plane
head is the one exception: it needs the 3-plane feature recipe and
inference.py does not drive it.

Use an axial head. Axial fluid-sensitive exists for 4,407/4,407 studies;
sagittal for only 4,150, so a sagittal model has no input for ~6% of them. The
script falls back to another plane for those and reports how many needed it.
