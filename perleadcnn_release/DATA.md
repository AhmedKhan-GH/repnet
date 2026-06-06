# Data

The model was developed on a clinical cohort of 12-lead ECGs from pregnant
patients at UC Davis Health. **The dataset is protected health information
(PHI) and is not included in this release.** It is available from the authors
only under an appropriate IRB approval and data-use agreement.

This document specifies the exact format the code expects so the pipeline can
be run against the original cohort (or any equivalently-formatted dataset).

## Expected layout

Point the code at the data directory via the `REPNET_DATA_DIR` environment
variable (default: `./data/seniordesign_upload`):

```
<data_dir>/
  metadata.csv            # one row per ECG recording
  ekg_data/
    <ECGTestID>.csv       # one file per recording (filename == ECGTestID)
    ...
```

## `metadata.csv`

Must contain at least these columns (others are ignored):

| Column          | Type   | Meaning                                                        |
|-----------------|--------|----------------------------------------------------------------|
| `ECGTestID`     | int    | Unique recording id; matches the `<ECGTestID>.csv` filename    |
| `Pat_Obfus_MRN` | string | De-identified patient id — used for **patient-grouped** splits |
| `PatLabel`      | string | Diagnosis label (see positive value below)                     |

A recording is labelled **positive (preeclampsia)** when `PatLabel` equals:

```
Preeclampsia or Other Hypertensive Disorders of Pregnancy
```

Everything else is labelled negative.

## `ekg_data/<ECGTestID>.csv`

One CSV per recording, with one column per lead (header row required), in any
order — the loader selects them by name:

```
I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6
```

Each column is a single lead's voltage time series. Accepted lengths:
- **5000 samples** (500 Hz), or
- **2500 samples**, which are resampled up to 5000 so filtering is identical.

## What the loader does (`src/data.py`)

1. Keep recordings whose `ECGTestID` has a matching CSV with all 12 leads.
2. Resample 2500-sample records to 5000 (500 Hz).
3. Per lead: 0.5 Hz high-pass (baseline-wander removal), 60 Hz notch, z-score.
4. Downsample ×2 → **2500 samples @ 250 Hz** (the model input).
5. Drop non-finite, flat-lead, and unlabelled-patient records.

After cleaning, the original cohort yields **2,178 recordings** from **1,383
patients**, **335 positive (15.4%)**.

## Splits

30 patient-grouped splits. For split `i`, the seed is `i*7 + 1000`; an outer
`StratifiedGroupKFold(n_splits=5)` defines the test fold and (for training) an
inner `StratifiedGroupKFold(n_splits=8)` defines validation. No patient appears
in more than one partition. This is implemented once in `src/data.py` and used
by both training and evaluation.
