Developed machine learning models to detect 12 clinically important abnormalities in knee MRI scans as part of the RSNA Knee MRI Abnormality Detection competition as part of the [Knee Abnormality Detection Challenge](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

## Scripts

### Main Evaluation

| Script | Description |
|---|---|
| `final_eval.py` | Compares image-only training with training augmented by report-derived labels using five-fold cross-validation. Results: `weak/final_cv_results.json`. |
| `milestone_head.py` | Evaluates the effect of increasing the number of report-derived labels on the 12-study evaluation split. |
| `inference.py` | Loads a trained model head and generates `submission.csv`. |

### Model Configurations

| Script | Description |
|---|---|
| `final_model.py` | Sagittal-plane model. |
| `final_model_axial.py` | Axial-plane model. |
| `final_model_b2.py` | Three-plane model. |

### Ablation Studies

| Script | Description |
|---|---|
| `lever_a.py` | Compares slice-aggregation methods. Results: `weak/lever_a_results.json`. |
| `lever_b.py` | Compares MRI-plane configurations and the contribution of report-derived labels. Results: `weak/lever_b_*.json`. |
| `lever_c.py` | Implements backbone fine-tuning. The script was smoke-tested, but the full experiment was not run. |
| `lever_d.py` | Evaluates a BiLSTM-based multi-plane model with per-plane attention. Results: `weak/lever_d_3plane.json`. |
| `lever_e.py` | Compares pretrained backbones using frozen feature extractors. Results: `weak/lever_e_backbones.json`. |

### Feature Extraction

| Script | Description |
|---|---|
| `extract_*.py` | Extracts and caches image features for downstream experiments. |

## Runtime

With the backbone frozen, only the classification head is trained. The table reports the number of epochs required to reach 99% of the final pooled out-of-fold (OOF) AUC, averaged over 3 random seeds using 5-fold cross-validation on 56 labeled studies.

| Configuration | Feature Dim. | Head Parameters | Epochs to 99% | Time/Epoch/Fold | Time to Convergence |
|:---|---:|---:|---:|---:|---:|
| Sagittal / ConvNeXt-Small | 768 | 201K | 5 | 0.12 s | ~0.6 s |
| Axial / ConvNeXt-Small | 768 | 201K | 7 | 0.12 s | ~0.8 s |
| Axial / ConvNeXt-Base | 1024 | 268K | 11 | 0.14 s | ~1.5 s |
| Three planes / ConvNeXt-Small | 2304 | 659K | 5 | 0.20 s | ~1.0 s |


*Note: Convergence times are approximate and depend on hardware and implementation details.*

## Peak AUC vs. Training Duration

Extending training to 80 epochs did not improve performance over the peak validation AUC for any configuration. The table compares peak out-of-fold (OOF) AUC with the AUC at epoch 80.

| Configuration | Peak AUC | Peak Epoch | AUC at Epoch 80 | AUC Change |
|:---|---:|---:|---:|---:|
| Sagittal / ConvNeXt-Small | 0.7396 | 37 | 0.7332 | -0.0064 |
| Axial / ConvNeXt-Small | **0.7748** | 29 | 0.7514 | -0.0234 |
| Axial / ConvNeXt-Base | 0.7715 | 58 | 0.7611 | -0.0104 |
| Three Planes / ConvNeXt-Small | 0.7714 | 15 | 0.7632 | -0.0082 |


## Trained Model Checkpoints

We provide trained model checkpoints from experiments on the RSNA Knee Abnormality Detection challenge. These experiments explore different imaging planes, backbone architectures, and label sources for predicting 12 knee abnormalities from MRI scans.

| Checkpoint | Description |
|---|---|
| [`head_3plane_weaklabels.pt`](checkpoints/head_3plane_weaklabels.pt) | Classification head trained using axial, sagittal, and coronal MRI images with additional weakly labeled training data. |
| [`head_axial_0751.pt`](checkpoints/head_axial_0751.pt) | Classification head trained using axial MRI images. |
| [`head_axial_convnext_base_0761.pt`](checkpoints/head_axial_convnext_base_0761.pt) | Classification head using a ConvNeXt-Base backbone with axial MRI images. |
| [`head_sagittal_0733.pt`](checkpoints/head_sagittal_0733.pt) | Classification head trained using sagittal MRI images. |
| [`head_sagittal_imagelabels_0656.pt`](checkpoints/head_sagittal_imagelabels_0656.pt) | Classification head trained using sagittal MRI images with image-level labels. |

*Note: Checkpoints contain classification heads only; inference requires the corresponding backbone and preprocessing.*


