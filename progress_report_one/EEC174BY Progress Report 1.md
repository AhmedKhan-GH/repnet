**Introduction**  
Currently, preeclampsia is diagnosed when a woman at least 20 weeks pregnant suffers from high blood pressure (140/90 mmHg) and another symptom, such as protein in their urine (proteinuria), a low blood platelet count, fluid in the lungs (pulmonary edema), etc. \[Preeclampsia\] It is an incredibly time-sensitive disorder, as preeclampsia can develop within days \[Early Signs of Preeclampsia\]. With an early diagnosis, preeclampsia can be prevented from further deterioration with proper monitoring, lifestyle changes, and pharmaceutical intervention \[AI-based preeclampsia detection and prediction with electrocardiogram data\]. However, there are many cases, especially in parts of the world with limited medical care, where the diagnosis is given after preeclampsia becomes more severe \[Preeclampsia And Eclampsia\]. It may even get to the point of eclampsia, where the disorder affects the brain function, causing seizures \[Preeclampsia And Eclampsia\]. There is no cure to preeclampsia until the baby is delivered \[Preeclampsia\]. Due to this, the ability to predict preeclampsia early on is crucial, as it allows women to take preemptive measures to combat the disorder. 

**Related/Previous Work/Literature Review**  
Decision tree-based models have proven to be successful in predicting preeclampsia, though none through the use of ECGs for inference. Bulez et al observed the use of a LightGBM model using data from blood tests, blood pressure, and other factors from patient EHRs with the conclusion that age and features from the blood test were most important for inference \[Artificial Intelligence in Early Diagnosis of Preeclampsia\]. In Zhang et al’s work, the XGBoost model was used on urine samples to determine seven compounds that were most important in predicting preeclampsia \[Development of a Urine Metabolomics Biomarker-Based Prediction Model for Preeclampsia during Early Pregnancy\].

In \[AI-based preeclampsia detection and prediction with electrocardiogram data\], preeclampsia can be predicted 30-90 days before diagnosis by using a modified ResNet CNN with an input of 1-dimensional raw ECG signals among 12 leads. Their discussion section mentions future improvements by either reducing leads or using a single-lead model (lead I), which can be mimicked via smart wearables. The latter is backed up with \[Feasibility of remote monitoring for fatal coronary heart disease using Apple Watch ECGs\], where a single-lead model is used to predict fatal coronary heart disease. 

However, a potential flaw in this article is whether it will remain accurate given diversity, as their dataset was limited to mostly African-American women. We must also examine whether using a single-lead model will benefit the project, assuming it is possible for preeclampsia detection, as calculations of certain features require more than one lead.   
Another useful article \[Does a Reduced ECG Lead Set Contain the Full 12-Lead ECG Information for Interpretation\] discusses lead reduction, using only leads I, II, V2, and V4 for diagnosing morphology interpretations. This is done by using deep learning neural networks directly with ECG waves, rather than our current method of feature extraction. It is unclear whether this would be useful for determining preeclampsia, as the diagnosis is not directly connected to lead readings as morphology interpretations are. However, it should be considered, as it prioritizes the same leads that can be used for certain calculations \[Visualising the Novosel Formula: Comments on Dahl and Berg’s A for the mean electrical axis of the heart\].

**Evaluation Criteria**  
	As a binary classification problem, the main metrics used in determining the model performance will be area under curve (AUC), accuracy, precision, recall, and F1-score. MUSE values of certain features are reported from the UCDH Echo dataset which will be compared to features created locally from Neurokit to determine accurate implementation. Once the featurizations have been implemented, feature importance analysis will be used to determine which features from the ECGs are most useful for prediction, ultimately pruning the features by analyzing tradeoffs from the performance metrics.

**Implementation and Results**  
	The Python library, Neurokit has methods for cleaning ECG signals along with extracting certain features. Through the use of these methods, we are able to implement the features in Table \_. Hana Shaik, our graduate mentor, has already implemented ST-T abnormalities and used these features in detecting RWMA. Testing occurred on the Nightingale ECG dataset to ensure that featurization was occurring. As another form of validation, we were able to compare our implementation to Hana’s implementation. After receiving the balanced UCDH Echo dataset, we reformatted our Nightingale code to be able to process individual csv ECG files instead of individual lead files. We had to drop three ECGs due to 0’s in some leads or NaNs after delineation.  
	The main issue as of the moment is figuring out accurate QRS Axis and T-axis calculations in order to also obtain the frontal QRS-T angle. Currently we have two implementations for the QRS axis. The first is the isoelectric method, which uses a way of estimation by pinpointing the isoelectric lead and going \+/-90° based on if the perpendicular pair has a positive or negative QRS value. The second is by using the equation: QRS-axis \= ± arctan(2 \* aVF3 \* I) \[Calculating The QRS Axis\], \[Visualising the Novosel Formula: Comments on Dahl and Berg’s A for the mean electrical axis of the heart\]. This method would give a specific number, which may not be necessary as in a real clinic, small degree differences are insignificant  \[Calculating The QRS Axis\].  

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
	With the UCDH Echo dataset and MUSE values now released, we will be able to compare Neurokit values from MUSE values and determine if there are any drastic differences. One issue is there is not a reported value for the QRS axis nor specific equation the MUSE developers used, so we will have to experiment through feature analysis if our implementations are important for inference.  An RF model will be initially used, but other decision tree models will be tested such as LightGBM and XGBoost to achieve best performance. After model selection, feature importance and feature distribution will be analyzed through 𝛘2 testing or principal component analysis (PCA). Because the dataset is imbalanced, we may employ SMOTE to train on a balanced dataset.

**References**  
\[1\] A. Bulez, K. Hansu, E.S. Cagan, A.R. Sahin, and H.O. Dokumaci, “Artificial Intelligence in Early Diagnosis of Preeclampsia”, March 2024\.  
\[2\] Y. Zhang, K. G. Sylvester, B. Jin, R. J. Wong, J. Schilling, C. J. Chou, Z. Han, R. Y. Luo, L. Tian, S. Ladella, L. Mo, I. Maric, Y. J. Blumenfeld, G. L. Darmstadt, G. M. Shaw, D. K. Stevenson, J. C. Whitin, H. J. Cohen, D. B. McElhinney, and X. B. Ling, “Development of a Urine Metabolomics Biomarker-Based Prediction Model for Preeclampsia during Early Pregnancy”, May 2023\.