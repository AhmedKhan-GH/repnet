**Related/Previous Work/Literature Review**  
Decision tree-based models have proven to be successful in predicting preeclampsia, though none through the use of ECGs for inference. Bulez et al observed the use of a LightGBM model using data from blood tests, blood pressure, and other factors from patient EHRs with the conclusion that age and features from the blood test were most important for inference \[1\]. In Zhang et al’s work, the XGBoost model was used on urine samples to determine seven compounds that were most important in predicting preeclampsia \[2\].

**Evaluation Criteria**  
	As a binary classification problem, the main metrics used in determining the model performance will be area under curve (AUC), accuracy, precision, recall, and F1-score. MUSE values of certain features are reported from the UCDH Echo dataset which will be compared to features created locally from Neurokit to determine accurate implementation. Once the featurizations have been implemented, feature importance analysis will be used to determine which features from the ECGs are most useful for prediction, ultimately pruning the features by analyzing tradeoffs from the performance metrics.

**Implementation and Results**  
	Without current access to the UCDH Echo dataset, implementing the ECG features for the RF model is what is being done. The Python library, Neurokit has methods for cleaning ECG signals along with extracting certain features. Through the use of these methods, we are able to implement the features in Table \_. Hana Shaik, our graduate mentor, has already implemented ST-T abnormalities and used these features in detecting RWMA. Testing occurred on the Nightingale ECG dataset to ensure that featurization was occurring. As another form of validation, we were able to compare our implementation to Hana’s implementation. (Want to include more after 4/14 meeting such as discussion of QRS-axis, T-axis)

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

**Future Work**  
	Once the UCDH Echo dataset and MUSE values are released, we will be able to compare Neurokit values from MUSE values and determine if there are any drastic differences. An RF model will be initially used, but other decision tree models will be tested such as LightGBM and XGBoost to achieve best performance. After model selection, feature importance and feature distribution will be analyzed through 𝛘2 testing or principal component analysis (PCA). Because the dataset is imbalanced, we may employ SMOTE to train on a balanced dataset.

**Acknowledgements**

**References**  
\[1\] A. Bulez, K. Hansu, E.S. Cagan, A.R. Sahin, and H.O. Dokumaci, “Artificial Intelligence in Early Diagnosis of Preeclampsia”, March 2024\.  
\[2\] Y. Zhang, K. G. Sylvester, B. Jin, R. J. Wong, J. Schilling, C. J. Chou, Z. Han, R. Y. Luo, L. 

Tian, S. Ladella, L. Mo, I. Maric, Y. J. Blumenfeld, G. L. Darmstadt, G. M. Shaw, D. K. Stevenson, J. C. Whitin, H. J. Cohen, D. B. McElhinney, and X. B. Ling, “Development of a Urine Metabolomics Biomarker-Based Prediction Model for Preeclampsia during Early Pregnancy”, May 2023\.

