# Phase 3 Experimental Review (Shelved)

## Section Text

### Phase 3 Literature Review - Experiment

#### ECG Classification Large Foundation Models

Large foundation models hold promise for their training on millions of ECG waveforms, they save us the trouble of access to limited data by training the earlier layers on massive amounts of data, giving it robust feature extraction that can then be reasoned about for different classification tasks after the fact. We selected Han et al. as the leading literature review on ECG Classification LFM's \cite{han_systematic_2025}. We have also selected ECG-FM as our LFM of choice since it is the leading open model in the field. They provide a GitHub repository with all relevant infrastructure and pipelines such as signal preprocessing and running benchmarks on public datasets \cite{mckeen_ecg-fm_2025}. We will compare our results to fine tuned ECG-FM.

---

## Table Rows (from Summary of Reviewed Literature)

### Phase 3 --- Experimental Review

**ECG Classification Large Foundation Models**

| Paper | Type | Reviewed | Purpose |
|---|---|---|---|
| Han et al. \cite{han_systematic_2025} | Review | ✓ | Review of large foundation models for ECG |
| McKeen et al. \cite{mckeen_ecg-fm_2025} | Original | | ECG-FM open source foundation model |

**ECG Classification Transformer Studies**

| Paper | Type | Reviewed | Purpose |
|---|---|---|---|
| Kim et al. \cite{kim_novel_2025} | Original | | |
| L. Liu et al. \cite{liu_hybrid_2025} | Original | | |

---

## Section Text for Kim et al. and L. Liu et al.

### ECG Classification Transformer Studies (partial)

For future reading we have selected Kim et al. as a leading paper in the application of transformers for ECG classification. It implements Stockwell transform, a function that modulates the size of the window based on the frequency, giving high frequencies better spacial locality, and low frequencies a more accurate frequency reading \cite{kim_novel_2025}. Stockwell transform localizes a signal's energy in both time and frequency rather than forcing it to learn those features from raw timeseries data or discarding it in total transforms like Fourier, enabling models like CNN to pick up on the movement of levels of energy as cardiac events from the visual distributions. We aim to utilizse approaches that trasnform 1-D to 2D representations to better learn features that need complex mathematics to expose them.

Stockwell transform equation:
S(t,f) = ∫ x(τ) * (|f|/√(2π)) * e^(-(t-τ)²f²/2) * e^(-i2πfτ) dτ

Figure: Stockwell transform representations of ECG heartbeat types (N, S, V, Q). Row 1: raw signal; Rows 2–4: magnitude, real, and imaginary components of the S-transform. Image: 41598_2025_92582_Fig3_HTML.png. Reproduced from Kim et al. \cite{kim_novel_2025}.

For future reading we also selected L. Liu et al. whom implements a 1-D CNN with Bidirectional Encoding Representations from Transformers (BERT). BERT is a bidirectional attention mechanism which captures both past and future context. Unlike traditional applications of transformers such as language models or video models that posit only past causal context for inference, the heart functions in such a way that the electrical or physical activity in one moment can cause knock on effects to the system, hence future context \cite{liu_hybrid_2025}.
