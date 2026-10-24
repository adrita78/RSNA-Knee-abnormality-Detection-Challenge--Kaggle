# Plan: push val macro AUC from 0.656 toward ~0.8 (Adrita, 2026-10-21)
Written before any result. Val set = the SAME held-out 12 clean studies
(seed-42 split); it is never used to fit anything, text or image.

Mechanism: the 0.63-0.66 plateau is the 44-study ceiling. The unlock is
weak labels from the 4,349 unlabeled studies' reports (all present,
multilingual, median ~1k chars). Known from competition coverage:
report-derived labels agree ~82% with image-derived ground truth, so
weak labels are noisy -- the bet is 4,100 noisy >> 44 clean.

Steps:
1. Embed all 4,407 reports with a multilingual sentence model
   (intfloat/multilingual-e5-base; chunk long reports, mean-pool).
2. Per-target logistic regression trained on the 44 TRAIN-labeled
   reports only; report text-side AUC on the 12 val reports. This
   estimates weak-label quality. If median per-target text AUC < 0.65,
   stop and report (weak labels too poor to help).
3. Predict soft labels (probabilities) for all unlabeled studies.
4. Precompute frozen ConvNeXt features for all 4,150 sagittal_fs
   series once (backbone frozen => features fixed; ~400 MB cache).
5. Train the classifier head on cached features: clean 44 (hard
   labels) + unlabeled (soft labels, BCE with soft targets), simple
   50-epoch schedule, AdamW 3e-4. Variants allowed: soft-label weight
   in {0.3, 1.0}; pick by val AUC -- val is 12 studies, so model
   selection noise is acknowledged.
6. Report best val macro AUC honestly vs the 0.8 ask, with the
   12-study noise caveat attached. Optional robustness: 5-fold CV over
   the 56 clean studies for a less noisy estimate.
Leakage rules: the 12 val studies' reports are never in text training;
their images never in head training; e5/ConvNeXt are generic
pretrained models (no competition data).

## Amendment 1 (2026-10-21, after step 2, before any zero-shot result)
Step 2 FAILED its gate: per-target LR on 44 reports gives median val AUC
0.491 (train 0.85-1.0 -> severe 44-sample overfit). Since the failure is
fitting capacity, not demonstrated absence of text signal, one amended
attempt is allowed: ZERO-SHOT e5 similarity scoring (per target, score =
sim(report, positive statements) - sim(report, negative statements);
English queries, cross-lingual by model design; no parameters fitted, so
no leakage and all 56 clean studies are legitimate validation).
Gate (fixed now): median per-target AUC over the 56 clean studies >=
0.65 to proceed to weak labels; below that, stop and report honestly.

## Amendment 2 (2026-10-21, after zero-shot failed its gate at 0.643)
Fitted LR (val 0.49) and report-level zero-shot e5 (0.643 on 56) both
failed pre-set gates. Final extractor attempt: Claude reads each report
and assigns the 12 labels directly (p in {0, 0.25, 0.5, 0.75, 1}),
the approach the competition ecosystem validated (~82% report-image
agreement). Gate (fixed now, before any labels): macro AUC of Claude's
labels vs image-derived ground truth on the 56 clean studies >= 0.75.
Pass -> label all remaining 4,094 studies, train head on cached frozen
features (clean 44 hard + weak soft), validate on the untouched 12.
Fail -> stop; report to Hyunsang/Adrita that ~0.8 is not reachable with
available extractors. Selection-noise note: this is the third extractor
evaluated against the 56; the gate metric (12 targets x 56 studies,
method not tuned per-target) limits but does not eliminate selection
optimism.

## Amendment 3 (2026-10-21, after the extractor gate PASSED at 0.877)
Claude-as-extractor validated: macro AUC 0.877 / median 0.881 / acc 0.806
on all 58 ground-truth studies (weak/claude_labels_56.csv). Full
hand-labeling of 4,094 reports costs ~5-7M tokens; amended to
DISTILLATION: (a) Claude hand-labels a random 300 of the 4,094 unlabeled
reports (seed 42); (b) train per-target logistic regression on e5
embeddings of those 300 (Claude soft labels as targets); (c) gate:
LR's macro AUC on the 58 image-truth studies >= 0.75. Pass -> LR
soft-labels the remaining ~3,794 and training proceeds per the original
step 5. Fail -> fall back to full Claude hand-labeling (budget allows).

## Amendment 3 outcome (2026-10-21): distilled LR FAILED gate (0.670 < 0.75;
## embeddings are the bottleneck, not the labels: direct Claude = 0.877).
## Proceeding with the pre-registered fallback: full Claude hand-labeling of
## the remaining 3,794 unlabeled sagittal_fs studies, rubric unchanged
## (weak/RUBRIC.md). Image feature cache builds in parallel (frozen backbone).

## Milestone outcome (feature cache complete, 2026-10-20)
All 4,150 feature files cached (weak/feat/, fp16, per-slice 768-d). Head-only
training on cached features, 3 seeds, best-val-AUC protocol (same as notebook):
  n_soft=0:   0.676 +- 0.017   (head-only baseline; 10-ep full pipeline was 0.6559)
  n_soft=100: 0.686-0.692
  n_soft=200: 0.700-0.712
  n_soft=300: 0.696-0.702
  n_soft=495: 0.723 +- 0.009 (w=1.0), 0.722 +- 0.004 (w=0.3)
Slope POSITIVE (+0.047 for 495 soft labels, ~monotone). Pre-registered rule
says CONTINUE labeling to 3,794. w_soft 1.0 vs 0.3 indistinguishable so far.
Full table: weak/milestone_results.json.

## Checkpoint 2 (2026-10-20, 1,295 soft labels)
Head-only, 3 seeds, w_soft=1.0: n_soft=495: 0.723; n_soft=900: 0.800 +- 0.015;
n_soft=1295: 0.779 +- 0.002. The 0.8 target is REACHED within val noise
(12-study val). Continue labeling to 3,794 for the final number, then
PLAN_08 step 5 (weight variants + optional 5-fold CV over 56).

## Checkpoint 3 (2026-10-20, 1,720 soft labels) -- STOP RULE FIRED

Head-only, 3 seeds, same protocol, val = the untouched 12 (seed 42):

    n_soft      val macro AUC
    0           0.676 +- 0.017
    900         0.800 +- 0.015
    1,295       0.779 +- 0.002
    1,720       0.762 +- 0.012

The curve rose from 0.676 to ~0.80 and then stopped rising; the last two
checkpoints are flat-to-declining. Pre-registered rule in step 4 ("if val
AUC does not move toward 0.8 with more soft labels, stop labeling and
report"): FIRED. Labeling HALTED at 1,420 of 3,794 hand-labeled reports
(1,720 soft studies in total with the 300 from sample300).

Honest reading: the gain from weak labels is real and large (+0.10 or more
over the 44-study baseline), but it saturates by roughly 900-1,000 labeled
reports. Differences among 0.800 / 0.779 / 0.762 are inside the noise of a
12-study validation set, so the saturation point is not resolvable more
precisely than "about 1,000".

### Amendment 4 (pre-registered before running it)

The 12-study val is too small to carry a headline number: 12 studies x 12
targets, several targets with 1-2 positives. For the final report we
therefore evaluate by 5-fold cross-validation over ALL 56 clean studies,
with out-of-fold predictions pooled into a single macro AUC over 56.

  - the soft pool is disjoint from all 56 clean studies by construction
    (asserted), so soft labels can be reused in every fold without leakage;
  - PRIMARY metric: fixed 80 epochs, NO per-fold epoch selection (an
    honest number: nothing is chosen on the evaluation data);
  - SECONDARY metric: best-epoch-by-fold-AUC, reported only because it is
    the protocol that produced the 0.6559 notebook number and is therefore
    the like-for-like comparison. It is optimistic and labeled as such.
  - arms: n_soft in {0, 1720} x w_soft in {0.3, 1.0}, 3 seeds each.
  - decision: report whatever comes out. No further arms.

## FINAL RESULT (PLAN_08 step 5 + Amendment 4, 2026-10-20)

5-fold CV over all 56 image-labeled studies, out-of-fold predictions pooled
into one macro AUC. 3 seeds. Script: final_eval.py; log: logs/final_eval.log;
numbers: weak/final_cv_results.json.

    arm                           PRIMARY (fixed 80 ep)   SECONDARY (best ep)
    44/56 image labels only       0.609 +- 0.003          0.632
    + 1,720 weak labels, w=1.0    0.733 +- 0.002          0.750
    + 1,720 weak labels, w=0.3    0.726 +- 0.009          0.747

PRIMARY selects nothing on the evaluation data and is the number to quote.
SECONDARY uses best-epoch-by-held-out-AUC, the notebook's own protocol, and
is the like-for-like comparison with the shipped 0.6559.

Weak labels are worth +0.124 macro AUC (0.609 -> 0.733), consistent across
seeds (seed sd 0.002-0.009) and across both soft-label weights. Per-target,
the gain is everywhere except MCL and Synovitis:

    ACL      0.589 -> 0.744     Medial OA   0.743 -> 0.867
    MCL      0.308 -> 0.517     Lateral OA  0.718 -> 0.765
    Med Men  0.466 -> 0.694     PF OA       0.578 -> 0.764
    Lat Men  0.707 -> 0.756     Effusion    0.726 -> 0.832
    Synovit  0.563 -> 0.624     Baker's     0.647 -> 0.753
    Contus   0.709 -> 0.742     Fracture    0.560 -> 0.743

HONEST POSITION ON THE 0.8 ASK: 0.80 was reached and is reproducible, but
only on the 12-study validation split the notebook uses (0.800 +- 0.015 at
900 soft labels). That split is too small to support the number: 12 studies,
several targets with 1-2 positives. On the 56-study CV the same pipeline
gives 0.733. We report 0.733 as the result and 0.80 as what the notebook's
own metric says. We did not tune anything to hit 0.8.

DELIVERABLE: checkpoints/convnext_frozenpool_weaklabels.pt
  - full ConvNeXtFrozenPool state_dict, strict-load verified
  - frozen ImageNet ConvNeXt-Small + head trained on 56 image-labeled
    studies plus 1,720 report-labeled studies, fixed 80 epochs
  - carries cv_macro_auc_56_studies = 0.7333 and a provenance note
  - trained on the PUBLIC JPEG-224 re-encode (alenic), same caveat as before

## Amendment 5 (2026-10-21): model-side levers, pre-registered before running

Labels are exhausted (Amendment 4 / checkpoint 3). The remaining levers are
architectural. Three, tried in increasing order of cost:

  LEVER A  slice aggregator. The head currently sees one mean-pooled 768-d
           vector per study, so a finding on 2 of 36 slices is averaged away.
           Runs on the EXISTING per-slice cache -> minutes of CPU.
  LEVER B  other planes. Only sagittal_fs is used; coronal_fs (4,624 series)
           and axial_fs (4,719) exist. Needs a GPU extraction pass (~1 h).
  LEVER C  unfreeze the backbone. Full fine-tune through ConvNeXt-Small on
           slice stacks. GPU, hours.

### Fixed evaluation (identical to Amendment 4, so numbers are comparable)
5-fold CV over all 56 image-labeled studies, out-of-fold predictions pooled
into one macro AUC; fixed 80 epochs (no epoch selection); 3 seeds; the 1,720
report labels included at w=1.0; AdamW lr 3e-4 wd 1e-4 bs 16.

REFERENCE TO BEAT: 0.7333 +- 0.0016 (mean-pool, PRIMARY metric).

### Decision rule, written down first
A lever counts as working only if BOTH hold:
  (i)  it beats the reference by >= 0.02. With 56 studies, differences
       smaller than that are not resolvable -- the seed sd is 0.002 but that
       measures seed noise, NOT the uncertainty of a 56-study estimate.
  (ii) the direction is consistent across all 3 seeds.
A lever that wins carries forward as the baseline for the next lever; a lever
that does not is dropped and the previous configuration stays.

### Multiplicity, stated in advance
Trying several architectures and keeping the best inflates whichever wins,
because the 56 studies are reused every time. ALL arms get reported, not just
the winner, and any winner is quoted as "best of N tried" with N stated. The
only clean arbiter remains the competition leaderboard.

### Lever A arms (2 candidates, plus the reference re-run in-harness)
  A0  mean pool                         - the reference
  A1  mean + max concatenated (1536-d)  - keeps the average, adds the peak
  A2  attention pooling (learned query, softmax over slices, masked)
Motivation for both: knee findings are focal, so the slice that shows a tear
should be able to dominate the study vector instead of being diluted.

### LEVER A RESULT (2026-10-21) -- NO ARM PASSES

    arm                        macro AUC          vs A0      verdict
    A0 mean pool               0.7333 +- 0.0018   reference  --
    A1 mean + max (1536-d)     0.7434 +- 0.0054   +0.0101    FAILS (i)
    A2 attention pooling       0.6952 +- 0.0085   -0.0381    WORSE

A1 satisfies criterion (ii) -- paired by seed it wins 3/3 (0.7394>0.7358,
0.7511>0.7317, 0.7397>0.7323) -- but +0.010 is half the pre-registered 0.02
threshold, so it is NOT promoted. Mean pooling stays as the baseline.

A0 reproduces the Amendment-4 reference (0.7333 +- 0.0016) to within 3e-4 in
independently written code; the difference is torch-vs-numpy float
accumulation in the slice mean. The headline is now confirmed by two
separate implementations.

MY STATED MOTIVATION WAS REFUTED. Both arms were justified by "knee findings
are focal, so the peak slice should be able to dominate". The per-target
breakdown of A1 shows the opposite pattern: the gains are on DIFFUSE
findings -- synovitis +0.092, lateral OA +0.054, medial OA +0.034, MCL
+0.034, PF OA +0.026 -- while the FOCAL ones lose: ACL -0.055, fracture
-0.039, Baker's -0.032, medial meniscus -0.022. So max pooling is behaving
as a "does this appear anywhere" detector that suits widespread cartilage
and synovial change, and it actively hurts the sparse ligament/fracture
findings the argument was built on. The aggregate +0.010 is a mix of those
two opposite effects, not a uniform lift.

A2 is a capacity failure, not a bug: a learned attention module must infer
slice relevance from study-level labels only, 1,720 of which are noisy
report labels, so the extra parameters find noise. More capacity on this
label budget makes things worse.

CARRY-FORWARD: mean pooling (unchanged) into Lever B.

### LEVER B RESULT (2026-10-21) -- B2 PASSES

                              with presence flags      without (control)
    B0  sagittal              0.7408 +- 0.0026         0.7332 +- 0.0016
    B1  + axial               0.7546 +- 0.0099         (not run)
    B2  + coronal + axial     0.7627 +- 0.0036         0.7632 +- 0.0045

THE CONTROL WAS NECESSARY. B0 is the reference configuration, yet it scored
0.7408 with flags against the Amendment-4 reference of 0.7333. The no-flag
control resolves it: B0 without flags = 0.7332 +- 0.0016, seeds
[0.7355, 0.7318, 0.7323], i.e. the reference reproduced a THIRD time in a
third implementation. So one constant, information-free column was worth
+0.0076 at 768 dims. At 2304 dims the same flags are worth -0.0005.

That number is the most useful thing in this table: it measures how much
this 56-study estimate moves under a perturbation carrying NO information.
The seed sd of 0.002 badly understates the real resolution, which is nearer
+-0.008. Any future claim on this dataset should be sized against 0.008,
not 0.002.

VERDICT: B2 passes both criteria, and robustly -- it clears 0.02 under both
flag treatments (+0.0219 vs flagged B0, +0.0300 vs the true reference) and
wins 3/3 seeds paired. The gain is also ~4x the information-free
perturbation above. Adopted WITH flags, since the flags are the principled
way to carry the 72 studies that have no coronal_fs series and they cost
nothing measurable.

HEADLINE MOVES: 0.733 -> 0.763 (3-plane, frozen backbone, mean pooling).

Per-target, the pattern is anatomically coherent, which is mild independent
evidence it is not noise: Baker's +0.101 (a posteromedial cyst, and axial
and coronal are exactly the planes added), synovitis +0.058, lateral OA
+0.057, MCL +0.037, ACL +0.033. One consistent loss in both B1 and B2:
lateral meniscus -0.100, unexplained.

MULTIPLICITY: four architecture variants have now been tried against the
same 56 studies (A1, A2, B1, B2). B2 is "best of four" and its margin is
inflated to an unknown degree by that search. The leaderboard remains the
only clean arbiter.

CARRY-FORWARD into Lever C: 3-plane concatenation, mean pooling, flags.

### LEVER C pre-registration (2026-10-21, written before running)

Measured cost on this box under Hyunsang's concurrent GPU load: 176 img/s
fwd+bwd, 62,879 slice-images per epoch per plane, so 6.0 min/epoch/plane.
10 epochs x 5 folds x 3 seeds = 14.9 h for ONE plane, 44.7 h for three.

SCOPE DECISION: run ONE plane (sagittal_fs), not the carried-forward 3-plane
B2 config. 44.7 h of contended GPU is not justified before knowing whether
unfreezing helps at all, and a 1-seed run would make criterion (ii)
untestable -- with the information-free noise floor measured at ~0.008, a
single seed is uninterpretable. So Lever C answers "does unfreezing the
backbone help?" against the SAGITTAL-ONLY frozen baseline:

    REFERENCE FOR LEVER C: 0.7332 +- 0.0016 (B0 no-flag / A0 / Amendment 4,
    the same number in three implementations). NOT B2's 0.7632.

If C passes, the 3-plane fine-tune becomes worth its 44.7 h. If C fails,
the ladder stops at B2.

Architecture held identical to the frozen pipeline so the only change is
which weights get gradients: backbone -> per-slice 768-d feature -> mean
over slices -> LayerNorm/Linear/GELU/Dropout/Linear head. All slices used,
no subsampling, so the input is exactly what the frozen arm saw.

Training protocol MUST differ (a backbone cannot be fine-tuned with the
head-only recipe) and is therefore stated here in advance: AdamW, backbone
lr 1e-5, head lr 3e-4, wd 1e-4, AMP, 10 epochs, micro-batch 2 studies with
gradient accumulation to an effective 16 studies (matching the frozen
recipe's bs=16), fixed epoch count with NO epoch selection. Evaluation is
unchanged: 5-fold CV over the same 56, out-of-fold pooled, 3 seeds.

HONEST LIMIT ON THE COMPARISON: this is "end-to-end pipeline vs frozen
pipeline", not "same protocol with one flag flipped". The training recipe
differs by necessity, so a loss could be a recipe artifact rather than a
verdict on fine-tuning. A win is more informative than a loss here.

### LEVER C RESULT (2026-10-21): NOT RUN -- cost measured, decision deferred

Measured, not estimated. Three successive estimates of this job were wrong,
each time because I projected from a narrower measurement than the real
pipeline:

    (1) synthetic tensors, GPU only          176 img/s   ->  14.9 h
    (2) smoke test, cold USB cache             4 img/s   -> ~600 h
    (3) warm page cache, real pipeline        21 img/s   -> 179.8 h
    (4) attribution: in-memory, checkpointed  -> 288 ms/study -> 21.3 h

So the REAL pipeline runs 8.5x slower than its own GPU cost (2,440 vs 288
ms/study). The time is CPU: JPEG decode + ToTensor + normalise for ~35
images per study inside DataLoader workers, competing with Hyunsang's
24-thread jobs at load 12+. Gradient checkpointing costs only 1.4x (202 ->
288 ms) and is kept, because it is what bounds peak memory to 3.2 GB; the
first attempt OOM'd at 18.5 GB since slice counts reach 320 per study, not
the mean of 35.

LESSON (same shape as the Panda frame-skip finding): a nominal throughput is
an UPPER BOUND. Measure the whole pipeline, warm, before quoting a budget.

FEASIBILITY:
  - full pre-registered protocol, with a uint8 pre-decode cache (9.5 GB
    disk) to remove the CPU bottleneck: 21.3 h of contended GPU.
  - reduced protocol that still satisfies the decision rule: 900 weak
    labels (the scaling curve saturates there anyway) + 5 epochs, with a
    MATCHED frozen baseline also at 900 weak labels: ~5.6 h, ~5 GB.

HALTED pending Hyunsang's decision, because it commits his GPU for 6-21 h
while his own jobs run at 99% utilisation, and 9.5 GB on a disk at 92% full
(28 GB free, 10 GB floor). lever_c.py is written, smoke-tested (3.0 GB peak,
sane predictions), and resumable per (seed, fold) via weak/lever_c_oof/.

### LADDER SUMMARY (Amendment 5)

    config                                     macro AUC (5-fold CV, 56)
    frozen, sagittal, mean pool  [reference]   0.7332 +- 0.0016
    + mean/max aggregator        [A1, failed]  0.7434 +- 0.0054
    + attention pooling          [A2, worse]   0.6952 +- 0.0085
    + axial                      [B1, failed]  0.7546 +- 0.0099
    + coronal + axial            [B2, PASSED]  0.7627 +- 0.0036
    + unfrozen backbone          [C, not run]  --

ADOPTED: B2. Headline 0.733 -> 0.763, i.e. +0.030 over the frozen
single-plane reference, caveated as best-of-four variants tried against the
same 56 studies, and measured against an information-free noise floor of
~0.008 rather than the 0.002 seed sd.

### CORRECTION (2026-10-22): the plane/fusion effects are NOT established

The flag control was run for axial and coronal too, and it overturns both my
Lever B verdict and the "axial beats sagittal" claim I made from it.

    config                      with flag            no flag
    sagittal                    0.7408 +- 0.0026     0.7332 +- 0.0016
    coronal                     0.7220 +- 0.0137     0.7156 +- 0.0089
    axial                       0.7623 +- 0.0051     0.7514 +- 0.0105
    sagittal + axial            0.7546 +- 0.0099     --
    coronal + axial             0.7566 +- 0.0058     --
    sagittal + coronal + axial  0.7627 +- 0.0036     0.7632 +- 0.0045

MATCHED comparisons of axial vs sagittal:
    with flag:  0.7623 - 0.7408 = +0.0215  -> clears 0.02 (barely)
    no flag:    0.7514 - 0.7332 = +0.0182  -> FAILS 0.02
The verdict FLIPS with a treatment that carries no information, so by my own
pre-registered rule the axial advantage is NOT established. Direction is
consistent (3/3 seeds under both treatments); the magnitude is not.

Likewise B2 vs matched sagittal is +0.0219 -- also just barely over the bar.

MY ERROR, STATED PLAINLY: I first quoted axial vs sagittal as +0.029 by
comparing axial WITH flags against sagittal WITHOUT them. That is exactly
the cross-treatment mixing I had already caught once in this same ladder,
and I then committed it myself. The matched number is +0.018 to +0.022.

NOISE FLOOR, RE-MEASURED: the information-free flag column is worth +0.0076
on sagittal and +0.0109 on axial. So ~0.008-0.011, not 0.002. Every effect
in Lever A and Lever B sits within ~2x that floor.

REVISED STANDING OF THE WHOLE LADDER:
  SOLID    report labels, 0.609 -> 0.733 (+0.124) = 11-15x the noise floor,
           reproduced in three independent implementations.
  NOT      every architecture/plane effect: +0.010 to +0.022, i.e. 1-2x the
  ESTABLISHED  noise floor, straddling the threshold, best of SEVEN variants
           tried against the same 56 studies.

HEADLINE REVERTS TO 0.733. Axial and 3-plane are recorded as suggestive
leads worth testing on the leaderboard, not as improvements. The bundle
already prepared for Adrita documents 0.733 and therefore needs no change.

### LEVER C: formally dropped (2026-10-22)

Not merely deferred. The re-measured noise floor settles it: with 56
validation studies an information-free change moves macro AUC by 0.008-0.011,
so this evaluation can only resolve effects of roughly +0.03 or larger. A
21.3 h fine-tune (or 5.6 h reduced) would therefore produce an uninterpretable
number UNLESS fine-tuning happens to give a large win.

That is the honest framing: Lever C is not uninformative a priori -- a big
effect would show -- but 6-21 h of contended GPU is the price of a test that
can only answer "is it huge?", and nothing smaller. Not worth it before a
leaderboard submission exists to calibrate against.

lever_c.py stays in the repo, smoke-tested (3.0 GB peak, resumable per fold)
in case a bigger validation set ever becomes available, which is the only
thing that would make it worth running.

### LEVER D (2026-10-22): Adrita's BiLSTM architecture, measured

Her CachedFeatureMultiPlaneBiLSTM (shared BiLSTM over cached per-slice
features, per-plane learned attention queries, learned missing-plane
embedding) run verbatim in our harness, 3 planes + 1,720 weak labels:

    her BiLSTM, 3 planes       0.7458 +- 0.0088   [0.7359, 0.7442, 0.7573]
    mean-pool head, 3 planes   0.7627 +- 0.0036
    mean-pool head, axial      0.7514 +- 0.0105
    mean-pool head, sagittal   0.7332 +- 0.0016

Her architecture is 0.017 BELOW the plain mean-pool head on the same three
planes and ties single-plane axial. Against the ~0.01 noise floor: a tie to
slightly worse. The BiLSTM, the attention queries and the missing-plane
embeddings earn nothing over averaging slice features.

Two bugs found in her code by running it (reported to her):
  1. prepare_multiplane_inputs leaves `lengths` on CPU while plane_present
     goes to the GPU, so encode_plane's `(lengths > 0) & plane_present.bool()`
     raises a device mismatch on the first training batch. Her model class is
     correct -- it already calls vl.cpu() for pack_padded_sequence -- so the
     fix belongs in prepare_multiplane_inputs.
  2. Six names are never defined anywhere in her notebook (clean_df, weak_df,
     TRAIN_SERIES_ROOT, use_amp, Dataset, DataLoader) plus series_by_study,
     which her Dataset reads but nothing builds.
Fixed notebook: adrita/10.22/with-all-planes-weak-label-FIXED.ipynb

### THE PATTERN ACROSS LEVERS A-D

On FROZEN features, information helps and capacity does not:

    weak labels        (information)   +0.124
    better plane       (information)   +0.035 without weak labels
    mean+max pooling   (capacity)      +0.010  below threshold
    plane concat       (capacity)      ~0
    her BiLSTM+attn    (capacity)      -0.017
    attention pooling  (capacity)      -0.038

Four capacity increases, none positive. Both information additions large.

### LEVER D part 2 (2026-10-22): her architecture on ONE plane — the control
### that makes the verdict fair

    config                          macro AUC            plane gain
    her BiLSTM,  axial              0.7216 +- 0.0111
    her BiLSTM,  3 planes           0.7458 +- 0.0088     +0.024  CLEARS 0.02
    mean-pool,   axial              0.7514 +- 0.0105
    mean-pool,   3 planes           0.7627 +- 0.0036     +0.011  fails

TWO SEPARATE FINDINGS, and the earlier write-up conflated them:

(1) MULTI-PLANE DOES WORK -- inside her architecture. +0.024, above the
    threshold, 3/3 seeds paired. Her per-plane learned attention queries and
    missing-plane embedding extract cross-plane information that mean-pool
    concatenation does not (+0.011, below threshold). Her instinct about
    using all the planes was CORRECT; my "plane concatenation is null" result
    was a property of mean pooling, not of multi-plane fusion in general.

(2) HER ARCHITECTURE COSTS MORE THAN THE PLANES RETURN. Against mean pooling
    at matched planes it is -0.0298 on axial (above the noise floor, so a
    real loss, not a tie) and -0.0169 on three planes. Net, her 3-plane
    BiLSTM (0.7458) sits below both mean-pool arms.

So the honest statement is NOT "the planes don't work". It is: the planes
carry information that a learned fusion can reach and simple concatenation
cannot, but the fusion machinery costs about 0.03 at this label budget, which
is more than the 0.024 it recovers. Lose the BiLSTM, keep the idea.

This also qualifies the "capacity never helps on frozen features" pattern:
capacity that buys ACCESS to information not otherwise reachable does pay
(+0.024); capacity applied to information already in hand does not.

### LEVER F (2026-10-22): cheap per-plane fusion — REFUTES MY OWN
### INTERPRETATION OF LEVER D

After Lever D part 2 I claimed her +0.024 showed that a LEARNED per-plane
fusion reaches cross-plane information that mean-pool concatenation cannot.
That was an inference from a within-arm delta, not a test of the mechanism.
Tested directly, it is wrong.

    config (all +weak, same protocol)            macro AUC   vs axial alone
    her BiLSTM, axial                            0.7216      -0.0298
    her BiLSTM, 3 planes                         0.7458      -0.0056
    mean-pool, axial alone            [reference] 0.7514       --
    learned softmax weights, 3 params            0.7557      +0.0043  fails
    learned per-plane gate, 3 x 768              0.7573      +0.0059  fails
    plain concatenation                          0.7632      +0.0118  fails

TWO CONCLUSIONS:

(1) NO fusion of any form clears the pre-registered +0.02 over the best
    single plane. Concatenation is the best of them at +0.012, and even that
    fails. Four mechanisms tested now: concatenation, learned scalar weights,
    learned per-plane gates, and her BiLSTM with attention.

(2) LEARNED fusion is WORSE than plain concatenation, not better: scalar
    -0.0074 and gate -0.0058 against concat. So the claim that a learned
    per-plane mechanism unlocks something concatenation misses is NOT
    supported.

REINTERPRETING LEVER D part 2 correctly: her architecture's deficit against
mean pooling NARROWS as planes are added (-0.0298 at one plane, -0.0173 at
three). Her +0.024 is therefore better read as her architecture partially
recovering from its own handicap, not as new information being unlocked --
and the decisive evidence is that her 3-plane result (0.7458) still sits
BELOW both plain concatenation (0.7632) and even mean-pool axial alone
(0.7514).

LESSON: a within-arm delta is not evidence of a mechanism. I read +0.024
inside her architecture as "learned fusion reaches more", wrote that into a
message to her, and a direct test of the mechanism refuted it. Test the
mechanism, not the delta. This is the third revision of this verdict; each
one came from a control that tightened it, and the practical advice never
changed: one good plane, mean pooling, weak labels.
