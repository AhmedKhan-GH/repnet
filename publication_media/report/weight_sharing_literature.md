# Weight Sharing & Channel-Independent Processing in Deep Learning

## Core Concept

In RepNet, a single 1D CNN is applied independently to each of the 12 ECG leads using **shared weights**. This means:
- The leads are processed in separate streams (channel-independent / no cross-lead mixing)
- The same convolutional parameters are reused across all 12 leads
- Each ECG recording effectively provides 12 training examples for the conv layers, multiplying the effective dataset size

This is a specific instance of a broader deep learning principle: **parameter sharing as implicit data augmentation and regularization**.

---

## Foundational Papers

### Weight Sharing / Parameter Tying

1. **Bromley et al. (1993)** - *"Signature Verification Using a Siamese Time Delay Neural Network"*
   - The original Siamese network paper. Two identical subnetworks with shared weights process two inputs independently, then compare representations.
   - Directly analogous to RepNet: same CNN applied to each lead independently, then combined via attention.
   - Introduced the principle that weight sharing across parallel streams acts as regularization and forces learning of input-agnostic features.

2. **Koch et al. (2015)** - *"Siamese Neural Networks for One-shot Image Recognition"*
   - Demonstrated that weight-shared twin networks excel in low-data regimes by effectively multiplying training signal.
   - Relevant to RepNet's data-limited setting (~335 PE samples).

3. **LeCun et al. (1998)** - *"Gradient-Based Learning Applied to Document Recognition"*
   - Foundational CNN paper establishing weight sharing (same kernel applied across spatial positions) as a core inductive bias.
   - RepNet extends this from spatial weight sharing (within a lead) to cross-lead weight sharing (same kernel applied to different leads).

### Depthwise / Channel-Independent Convolutions

4. **Chollet (2017)** - *"Xception: Deep Learning with Depthwise Separable Convolutions"*
   - Introduced depthwise separable convolutions: per-channel convolutions (channel-independent) followed by pointwise mixing.
   - RepNet's architecture is conceptually similar: per-lead conv (depthwise-like) followed by cross-lead attention (pointwise-like mixing).

5. **Howard et al. (2017)** - *"MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"*
   - Popularized depthwise separable convolutions for parameter efficiency.
   - Same principle: process each channel independently with shared or separate kernels, then mix.

### Channel Independence in Time Series / ECG

6. **Nie et al. (2023)** - *"A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"* (PatchTST)
   - Introduced **channel-independent patching** for multivariate time series: each channel (variable) is processed by the same Transformer independently.
   - Showed channel independence with shared weights outperforms channel-mixing approaches, especially when channels have similar temporal patterns but different scales/offsets --- exactly the case with ECG leads.
   - Directly supports RepNet's design rationale.

7. **Zeng et al. (2023)** - *"Are Transformers Effective for Time Series Forecasting?"*
   - Showed that simple linear models with channel-independent processing often outperform complex architectures.
   - Reinforces that channel independence is a strong inductive bias for multivariate time series.

8. **Han et al. (2024)** - *"The Capacity and Robustness Trade-off: Revisiting the Channel Independent Strategy for Multivariate Time Series Forecasting"*
   - Systematic study of when channel-independent processing helps vs. hurts.
   - Finds it helps most when: (a) inter-channel relationships are weak, (b) data is limited, (c) channels share similar temporal dynamics.
   - All three conditions hold for RepNet's preeclampsia ECG dataset (Frobenius norm = 1.026, ~335 PE samples, same heartbeat morphology across leads).

### ECG-Specific Multi-Lead Architectures

9. **Butler et al. (2024)** - *"AI-based preeclampsia detection and prediction with electrocardiogram data"* (ECG-AI)
   - The baseline this project improves upon. Uses channel-**dependent** processing: all 12 leads fed as 12 input channels to a shared ResNet.
   - Leads mix from the first conv layer, losing lead identity.
   - RepNet's channel-independent approach was motivated by the limitations of this design.

10. **Joshi et al. (2024)** - RWMA classification from ECG
    - Bicameral approach (CNN + RF) that informed RepNet's dual-model design.

11. **Jiang et al. (2024)** - *"DCETEN: A lightweight ECG automatic classification network"*
    - Uses depthwise separable convolutions with efficient channel attention for ECG.
    - Similar philosophy: separate per-channel feature extraction + lightweight cross-channel mixing.

---

## Why It Works for ECG

The 12-lead ECG has a specific structure that makes channel-independent weight sharing ideal:

| Property | Implication |
|----------|-------------|
| All leads record the same heartbeat | Same temporal patterns (QRS, T-wave) appear in every lead |
| Leads differ in amplitude/polarity | Differences encode cardiac axis, not different phenomena |
| Inter-lead correlation is class-agnostic (Frobenius norm = 1.026) | Discriminative signal is within-lead, not between-lead |
| Small dataset (~335 PE samples) | Weight sharing gives 12x effective training data |

---

## Summary

The key papers to cite for justifying RepNet's weight-shared channel-independent design:

- **Siamese networks** (Bromley 1993, Koch 2015) --- weight sharing across parallel streams
- **PatchTST** (Nie 2023) --- channel-independent processing for multivariate time series, directly analogous
- **Han et al. 2024** --- theoretical/empirical analysis of when channel independence works (low data, weak inter-channel coupling, shared dynamics)
- **Xception/MobileNets** (Chollet 2017, Howard 2017) --- depthwise separable convolutions as the vision equivalent
- **Butler et al. 2024** --- the channel-dependent baseline that RepNet improves upon
