**Evaluation Criteria**  
	As a binary classification problem, the main metrics used in determining the model performance will be AUROC, AUPRC, accuracy, precision, sensitivity, specificity, and F1-score. MUSE values of certain features are reported from the UCDH Echo dataset which will be compared to features created locally from Neurokit to determine accurate implementation. Once the featurizations have been implemented, feature importance analysis will be used to determine which features from the ECGs are most useful for prediction, ultimately pruning the features by analyzing tradeoffs from the performance metrics. Current possible analyses include chi-squared testing (𝛘2), recursive feature elimination (RFE), principal component analysis (PCA) and Kolmogorov-Smirnov testing (KS).

**Implementation and Results**  
	The Python library, Neurokit has methods for cleaning ECG signals along with extracting certain features. Through the use of these methods, we are able to implement the features in Table \_(1). Hana Shaik, our graduate mentor, has already implemented ST-T abnormalities and used these features in detecting RWMA. Testing occurred on the Nightingale ECG dataset to ensure that proper featurization was occurring. As another form of validation, we were able to compare our implementation to Hana’s implementation. After receiving the balanced UCDH Echo dataset, we reformatted our Nightingale code to be able to process individual csv ECG files instead of individual lead files.   
	The main issue as of the moment is figuring out accurate QRS Axis and T-axis calculations in order to also obtain the frontal QRS-T angle. Currently we have two implementations for the QRS axis. The first is the isoelectric method, which uses a way of estimation by pinpointing the isoelectric lead and going \+/-90° based on if the perpendicular pair has a positive or negative QRS value. The second is by using the equation: QRS-axis \= ± arctan(2 \* aVF3 \* I) \[Calculating The QRS Axis\], \[Visualising the Novosel Formula: Comments on Dahl and Berg’s A for the mean electrical axis of the heart\]. This method would give a specific number, which may not be necessary as in a real clinic, small degree differences are insignificant  \[Calculating The QRS Axis\].  Dr. Ebong, a UC Davis Health cardiologist, is currently evaluating the accuracy of our implementations.

| Features | Equations |
| :---- | :---- |
| Pmax | max(max(P offsets \- P onsets) for 12 leads) |
| Pmin | min(min(P offsets \- P onsets) for 12 leads) |
| P-Wave Dispersion | Pmax \- Pmin |
| QT Dispersion | QTmax \= max(max(T offsets \- R onsets) for 12 leads) QTmin \= min(min(T offsets \- R onsets) for 12 leads) Qtmax \- Qtmin |
| Tp-e Interval | avg(T offsets \- T peaks) |
| Tp-e/QT | (Tp-e Interval) / (avg(T offsets \- R onsets) |
| Tp-e/QTc | Tp-e/QT \* sqrt(diff(R Peaks)) |
| ST Elevation | 1 if \> 50% elevated 0 otherwise |
| ST Depression | 1 if \> 50% depressed 0 otherwise |
| Avg. Amp. Diff. | avg(ST Amp. \- PR Amp.) |
| Med. Amp. Diff. | median(ST Amp. \- PR Amp.) |
| Max. Amp. Diff. | max(ST Amp. \- PR Amp.) |
| Min. Amp. Diff. | min(ST Amp. \- PR Amp.) |
| QRS Axis | TBD |
| T Axis | TBD |
| Frontal QRS-T Angle | abs(QRS Axis \- T Axis) |

Table \_(1) Neurokit Implementations

	Before training a random forest model on our datasets, it would be beneficial to extract novel MUSE features that correlate with the Neurokit features to determine if Neurokit performs well. Specific ECGs were dropped in order to obtain Neurokit featurizations for both balanced and unbalanced ECG waveform datasets with the detailed specifics in Table \_(2). In order to stay consistent, the MUSE featurization would also drop the same ECGs and compare only features that can possibly be extracted from their respective metadata csvs. This would result in not being able to compare the amplitude differential features in model performance due to the inability to obtain the PR amplitudes from the metadata csvs. MUSE featurization implementation will be listed in Table \_(3). After observing the average percent differences between the Neurokit and MUSE features, Neurokit features were consistently less than MUSE.

| Dataset | Original Total (Preeclamptic/Normal) | Dropped Patients (Preeclamptic/Normal) | Final Total (Preeclamptic/Normal) |
| :---- | :---- | :---- | :---- |
| Unbalanced | 337/1854 | 8/24 | 329/1830 |
| Balanced | 187/182 | 6/1 | 181/181 |

Table \_(2) ECG waveform details

| Features | Equations |
| :---- | :---- |
| Pmax | max(P\_Duration for 12 leads) |
| Pmin | min(P\_Duration for 12 leads) |
| P-wave Dispersion | Pmax-Pmin |
| QT Dispersion | QT \= Q\_Duration \+ R\_Duration \+ S\_Duration \+ R\_PrimeDuration \+ S\_PrimeDuration \+ T\_Duration \+ T\_PrimeDuration \+ (⅛ \* RR) per lead QT Dispersion \= max(QT) \- min(QT) |
| Tp-e Interval | T\_Duration \+ T\_PrimeDuration \- STEtoTPeak |
| Tp-e/QT | Tp-e Interval / QT |
| Tp-e/QTc | Tp-e/QT \* sqrt(RR) |
| ST Elevation | (BITFLGS & 8\) \!= 0 |
| ST Depression | (BITFLGS & 4\) \!= 0 |
| QRS Axis | RAxis |
| T Axis | TAxis |
| Frontal QRS-T Angle | abs(QRS Axis \- T Axis) |

Table \_(3) MUSE Implementations

	At the moment, Neurokit did not seem reliable in extracting features from raw ECG waveforms, but after passing the extracted featurization datasets into a random forest model, we get a clearer understanding. The setup of the model consists of a Stratified Group K Fold to split the dataset into test and training sets based on an obfuscated patient MRN, which prevents data leakage. The results displayed in Table \_(4) reveal that across both the unbalanced and balanced datasets that the Neurokit and MUSE featurizations perform similarly. This enforces the idea that although Neurokit implementations underreport values, it does it consistently which does not affect the model too greatly.

| Dataset Featurizations | AUROC | AUPRC | Specificity | Sensitivity | Accuracy | Precision | F1-Score |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| Balanced (Neurokit) | 66.22 | 67.53 | 66.26 | 57.49 | 61.87 | 62.99 | 59.98 |
| Balanced (MUSE) | 62.47 | 65.66 | 60.27 | 59.73 | 59.95 | 60.22 | 59.75 |
| Unbalanced (Neurokit) | 63.19 | 22.62 | 63.83 | 54.73 | 62.44 | 21.33 | 30.67 |
| Unbalanced (MUSE) | 64.25 | 24.68 | 64.75 | 55.33 | 63.32 | 22.03 | 31.44 |

Table \_(4) Neurokit and MUSE extracted feature performance

	In addition to the previous experiment, another experiment implemented was to add the novel features extracted to the original metadata csvs to expand the features. When running this larger feature dataset containing 721 features for the metadata with extracted MUSE or 723 features when using extracted Neurokit, the results improve as seen in Table \_(5). Although the results improved, a preliminary experiment ran on just the metadata, achieved better performance which hints that the current model is overfitting. Chi-squared testing was performed to see if there is a relationship between the outcome and certain features, which show that AVL ST depression, V4 ST Depression, and V3 ST elevation were able to appear in the top 10 lowest p-values once across the four datasets. This is significant because it shows that our extracted features do make a difference in prediction.

| Dataset Featurizations | AUROC | AUPRC | Specificity | Sensitivity | Accuracy | Precision | F1-Score |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| MUSE (csv) \+ Neurokit balanced | 70.37 | 72.94 | 67.92 | 63.89 | 65.85 | 67.57 | 65.39 |
| MUSE (csv) \+ MUSE (extracted) balanced | 72.03 | 73.90 | 70.10 | 62.78 | 66.38 | 68.21 | 65.27 |
| MUSE (csv) \+ Neurokit unbalanced | 69.93 | 32.55 | 73.74 | 56.12 | 71.03 | 27.97 | 37.20 |
| MUSE (csv) \+ MUSE (extracted) unbalanced | 69.78 | 34.12 | 73.19 | 57.32 | 70.74 | 28.18 | 37.66 |

Table \_(5) Neurokit/MUSE \+ Metadata feature performance

**Future Work**  
	Feature engineering is the next step to obtaining a better model. Statistical methods will need to be employed to eliminate features that are closely related and features that are shown to sway the current models. There are multiple feature importance tests as discussed previously, but for the near future we will be focusing on chi-squared and RFE.  Because one of the datasets are imbalanced, we may employ SMOTE to achieve a much larger balanced dataset.

