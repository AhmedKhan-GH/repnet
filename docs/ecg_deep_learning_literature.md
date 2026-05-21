# ECG Deep Learning for Preeclampsia/Hypertensive Disorders -- Literature Summary

## 1. Preeclampsia Detection from ECG

**Butler, Gunturkun, Akbilgic et al. (2024)** -- "AI-based preeclampsia detection and prediction with electrocardiogram data," Frontiers in Cardiovascular Medicine. Used a modified ResNet CNN on 1D 12-lead ECG signals (~900 ECGs from UTHSC). Results: AUC 0.85 (0.77-0.93) on holdout; AUC 0.81 (0.77-0.84) on external validation (AHWFB). Prediction before diagnosis: AUC 0.92 at 30 days, 0.89 at 60 days, 0.90 at 90 days. Early-onset PE (<34 weeks): AUC 0.98. Five-fold cross-validation with 80/20 train/holdout split.

**Adedinsewo et al. (2021)** -- "Detecting cardiomyopathies in pregnancy and the postpartum period with an ECG-based deep learning model," European Heart Journal - Digital Health. Mayo Clinic group. Not PE-specific but demonstrated ECG-AI for pregnancy-related cardiac screening. AI-ECG AUC 0.94; AI-stethoscope AUC 0.98 for left ventricular systolic dysfunction.

## 2. Small-Dataset ECG Classification (<3000 samples)

**Transfer learning** is the primary strategy. A 2024 empirical study (arXiv 2402.02021) found fine-tuning is preferable for small downstream datasets, with CNNs benefiting more than RNNs. Two paradigms: (a) pretrain on large ECG corpus then fine-tune, (b) pretrain on ImageNet with ECG-to-image conversion. Key finding: fine-tuning improvement declines as dataset grows; training from scratch matches given enough data and time. **Domain adaptation** using labeled source + unlabeled target data also effective for small imbalanced ECG sets.

## 3. Architecture Choices for 1D ECG

- **ResNet1D**: 1D convolutional deep residual networks are baseline workhorses; used in the Butler et al. PE study.
- **InceptionTime**: Multi-scale parallel convolutions; strong benchmark performance on time-series tasks.
- **IncepSE (2023, SOICT)**: Combines InceptionTime with Squeeze-and-Excitation (SE) blocks, adding channel-wise attention to weight important features per lead/filter.
- **Depthwise separable convolutions**: Reduce parameters dramatically. Binarized DSCNN with merged convolution-pooling (2024, Sensors) enables efficient multi-class ECG classification. Inspired by MobileNet-V3, integrates SE blocks for lightweight yet accurate models.
- **Multi-scale feature extraction**: Multiple kernel sizes capture both local morphology and global rhythm; information bottleneck-based multi-scale networks (ScienceDirect, 2024).

## 4. Data Augmentation for ECG

Per a 2023 systematic survey (Sensors, PMC10256074) and 2025 comparative analysis (STAR):
- **Noise injection**: Gaussian noise added to random intervals (partial white noise).
- **Time warping**: Stretch/compress random ECG segments along the time axis.
- **Amplitude scaling/jitter**: Random amplitude perturbations.
- **Lead dropout**: Set a random lead's signal to zero; simulates missing-lead robustness.
- **Cutout**: Zero out a random time interval.
- **Mixup**: Interpolate between training samples and labels.
- **STAR (2025)**: Sinusoidal Time-Amplitude Resampling emerged as the most reliable single augmentation, delivering consistent AUROC gains.

## 5. Cross-Lead Attention / Multi-Lead Fusion

- **MLBF-Net**: Multi-Lead-Branch Fusion Network with lead-specific branches learning diversity, cross-lead concatenation learning integrity, and multi-loss co-optimization.
- **Adaptive multi-channel graph neural networks (2025)**: Model inter-lead spatial dependencies as graphs for 12-lead fusion.
- **Lead grouping (L5G-Net)**: Orthogonal lead-selection strategy groups complementary leads, reducing redundancy among the 12 standard leads.
- **EfficientECG (2025)**: Cross-attention with feature fusion for efficient 12-lead classification.

## 6. Regularization for Small Clinical Datasets

- **Focal loss**: Addresses class imbalance by down-weighting easy examples. Used in DSCNN ensembles for arrhythmia (PMC9481402) and advanced 1D-CNN frameworks with attention (2025).
- **L2 regularization / weight decay**: Standard; combined with focal loss in hybrid architectures.
- **Label smoothing**: Softens hard targets to prevent overconfident predictions.
- **Dropout**: Applied in convolutional and attention layers.
- **Ensemble / bagging**: Voting across multiple models improves generalization on small sets.
- **Mixed precision training**: Reduces memory footprint, enabling larger effective batch sizes.

## References

- Butler et al. 2024 -- PE detection from ECG (Frontiers in Cardiovascular Medicine)
- Adedinsewo et al. 2021 -- Cardiomyopathy in pregnancy (European Heart Journal - Digital Health)
- Transfer Learning in ECG Diagnosis: Is It Effective? (2024, arXiv 2402.02021)
- IncepSE: InceptionTime + SE blocks (2023, SOICT)
- Systematic Survey of ECG Data Augmentation (2023, Sensors, PMC10256074)
- STAR Comparative Analysis (2025, arXiv 2510.24740)
- Binarized DSCNN for ECG (2024, Sensors)
- MLBF-Net: Multi-Lead-Branch Fusion (arXiv 2008.07263)
- Multi-scale GNN for 12-lead ECG (2025, CMPB)
- EfficientECG: Cross-Attention (2025, arXiv 2512.03804)
- Deep CNN Ensemble with Focal Loss (2022, PMC9481402)
- 12-lead ECG Multi-Scale Hierarchical CNN (2025, Nature Scientific Reports)
