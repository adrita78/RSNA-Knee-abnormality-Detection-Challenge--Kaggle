# RSNA-Knee-abnormality-Detection-Challenge--Kaggle

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

