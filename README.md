# RSNA-Knee-abnormality-Detection-Challenge--Kaggle
https://www.kaggle.com/competitions/rsna-knee-abnormality-detection
# Handoff — RSNA 2026 knee abnormality detection

Everything from the 2026-10-20 to 10-22 work, self-contained.

    REPORT.md            what moved the number and what didn't. Start here.
    REPRODUCIBILITY.md   data, protocol, scripts, and the two traps that cost
                         real time. Read section 5 before trusting any margin.

    notebooks/
      single-plane-weak-labels.ipynb        the 0.733 pipeline, written up and
                                            executed cell by cell locally
      with-all-planes-weak-label-FIXED.ipynb  Adrita's notebook with the five
                                            blockers removed (see below)
    labels/
      weak_labels_by_study.csv              1,720 report-derived labels,
                                            keyed by StudyInstanceUID
      RUBRIC.md                             scoring rules, fixed before
                                            labeling began
    checkpoints/            ranked by CV macro AUC over the 56 labelled studies
      head_3plane_weaklabels.pt             0.7632  best measured, but 3 planes
                                            and NOT driveable by inference.py
                                            (needs the 3-plane feature recipe,
                                            documented inside the file)
      head_axial_convnext_base_0761.pt      0.7611  best SINGLE-PLANE config;
                                            ties the 3-plane arm on one plane
      head_axial_0751.pt                    0.7514  ConvNeXt-Small axial
      head_sagittal_0733.pt                 0.7333  the documented, verified
                                            pipeline; matches the notebook
      head_sagittal_imagelabels_0656.pt     original image-labels-only model

    For a submission use head_axial_convnext_base_0761.pt (one plane, 100%
    coverage, top of the single-plane table) or head_axial_0751.pt if you want
    the ConvNeXt-Small pipeline the notebook documents. The margins between
    0.7514, 0.7611 and 0.7632 are all at or below the noise floor, so pick on
    cost and coverage rather than on those numbers.
    scripts/            every experiment, one file per lever, plus inference.py
    results/            the JSON each script wrote, plus the pre-registration
                        and all amendments (PLAN_08_prereg_and_amendments.md)
    logs/               stdout of each run

## The checkpoints are 0.8 MB, not 199 MB

The frozen backbone in each is bit-identical to the stock torchvision weights
it names — verified tensor by tensor for ConvNeXt-Small, where the rebuilt
model reproduces the original forward pass to max|diff| exactly 0.0. Only the
~0.2M-parameter head is ours. Each head records its own backbone, and
`scripts/inference.py` builds whichever one it names, then writes a
submission.csv. Verified end-to-end with both a ConvNeXt-Small and a
ConvNeXt-Base head.

## One-line summary

Report-derived weak labels take the model from 0.609 to 0.733 macro AUC
(+0.124, the one solid effect). The axial plane is worth a further +0.035
before those labels are added and ~+0.018 after. Six attempts to add model
capacity on frozen features returned nothing or hurt. Plane fusion was tested
four ways — concatenation, learned scalar weights, learned gates, and a
BiLSTM with attention — and none beats the best single plane by the +0.02
needed to be detectable; the learned mechanisms are worse than plain
concatenation. Nothing below about +0.02 is measurable on 56 validation
studies, so a leaderboard submission is the only way to settle the smaller
leads.

## What was fixed in Adrita's notebook

    1  caching covered 14,010 series and never finished -> restricted to the
       1,778 studies actually used; ~8x less work with axial only
    2  series_by_study read by her Dataset but never built -> built
    3  six names never defined anywhere -> defined and imported
    4  lengths left on CPU while plane_present went to the GPU, so
       encode_plane died on the first training batch -> lengths moved to DEVICE
    5  11-study validation split -> 5-fold CV over all 58 labelled studies

Her model class, label-merge logic, Dataset and collate are unchanged; they
are correct. Her architecture was then run verbatim and measured at
0.7458 +- 0.0088 at three planes and 0.7216 +- 0.0111 at one — below a plain
mean-pool head in both cases, by 0.017 and 0.030 respectively (REPORT.md
section 4).

## Verification done before shipping

    - 21 of 21 numbers quoted in REPORT.md checked programmatically against
      the JSON files in results/
    - the headline reproduced in three independently written implementations,
      agreeing to 3e-4
    - the weak-label CSV re-keyed from row indices to StudyInstanceUID and
      verified to reproduce 0.7333 +- 0.0016 exactly
    - plane feature caches recomputed from the JPEGs and compared
      tensor-by-tensor: cosine 1.000000
    - both notebooks: every code cell parses, no undefined names, nothing used
      before definition (the check that found the six in her original)
    - inference.py run end-to-end on the public test sample, both with and
      without the missing-plane fallback

## Known limits

    - all figures come from the public JPEG-224 re-encode, not original DICOMs
    - 1,720 training labels were read off reports by a model, not images
    - thirteen configurations were tried against the same 56 studies, so any
      "best" among them is optimistically biased by that search
    - the competition test set is hidden; nothing here has been scored
