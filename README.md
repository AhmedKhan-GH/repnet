# RepNet — Preeclampsia Prediction from 12-Lead ECG

EEC 174 Senior Design Project, UC Davis

**Ahmed Khan, Lawrencedel Legaspi, Selena Phu**

RepNet is a multi-stage convolutional neural network with cross-lead attention for predicting preeclampsia (PE) from 12-lead electrocardiogram signals. The model processes all 12 ECG leads through three hierarchical convolutional stages, each followed by multi-head cross-lead attention, enabling the network to capture inter-lead repolarization patterns associated with hypertensive disorders of pregnancy.

## Public Materials

### Final Training Log

The complete output of the final 30-seed training run is in `final_data/repnet_crosslead_deeper_multiseed_pe_2026-05-04_21-06-34/`, including:

- `summary.txt` — Aggregate performance metrics across all seeds
- `results.log` — Full training execution log
- `config.json` — Hyperparameters and training configuration
- `results.json` / `per_seed_history.json` — Per-seed metrics and epoch-level history
- Interactive HTML visualizations (training curves, AUPRC distributions)

### Final Notebook

`notebooks/final_model/pe_model_analysis.ipynb` contains the detailed analysis of the final trained model. Additional notebooks in `notebooks/final_model/` cover exploratory data analysis (`eda.ipynb`) and meta-analysis across training runs (`meta_analysis_pe.ipynb`).

### Source Code

All model definitions, training scripts, and preprocessing pipelines are in `src/`:

- **Models** (`src/models/`) — RepNet variants, ResNet baseline, InceptionTime, and earlier cross-lead iterations
- **Training scripts** (`src/train_*.py`, `src/optimize_*.py`) — Full history of training and Optuna hyperparameter optimization runs across all architecture iterations
- **Data pipeline** (`src/data/`, `src/preprocessing/`) — ECG loading, filtering, augmentation, normalization, and class-balancing utilities
- **Evaluation** (`src/evaluation/`) — AUROC, AUPRC, Brier score, and sensitivity-at-specificity metrics
