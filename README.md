## Scripts and Reproducibility

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

### Reproducibility

Experimental results are recorded in JSON files alongside the corresponding scripts and logs. See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for details on the evaluation protocol, experiment configurations, and result files.
