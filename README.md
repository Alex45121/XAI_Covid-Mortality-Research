# XAI for COVID-19 Mortality Prediction

Course project for Explainable AI at the University of Twente, by Airidas Radzvilas and Alexander Kralev.

We predict whether a COVID-19 patient dies from three blood biomarkers, then test how good the explanations of those predictions really are. The main question is whether the explanation changes when you represent the same data in a different way.

## Data

The Tongji Hospital (Wuhan) dataset released with Yan et al. (2020), "An interpretable mortality prediction model for COVID-19 patients", Nature Machine Intelligence.

- Training set: 375 patients, about 6,000 measurement rows
- Test set: 110 patients, of whom 13 died
- Only three biomarkers are in both files: LDH, CRP and lymphocyte %, so the whole project uses just these three

The data files are not included in this repo. Download them from the paper's release and put them next to the notebook.

## Approach

We looked at the same data in two ways.

**Tabular.** Every patient becomes one row of 30 features (first, last, min, max, mean, slope and so on for each biomarker). On these we train logistic regression, a depth-3 decision tree and XGBoost, and explain them with SHAP and LIME.

**Temporal.** Every patient stays a sequence of measurements. An LSTM is trained on the raw sequences and explained with Gradient x Input and WindowSHAP.

We compared the explanations on four properties from the CO-12 framework (Nauta et al., 2023):

| Property | Test |
|---|---|
| Correctness | remove the top features and see how much the prediction drops, compared with removing random features |
| Continuity | add small noise to a patient and check if the explanation stays the same |
| Compactness | how many features you need to cover 80% of the explanation |
| Coherence | does the explanation match what is known medically |

We also checked for a Clever-Hans effect (the model learning how often a patient was tested instead of the actual values), looked at explanations per patient group, and tested how early a prediction is possible by only using the first 24, 48 or 72 hours of data.

## Results

| Model | ROC-AUC | Top biomarker in the explanation |
|---|---|---|
| XGBoost (tabular) | 0.994 | LDH |
| LSTM (temporal) | 0.969 | Lymphocyte % |

- Both models predict about equally well, but their explanations disagree. The tabular model mostly uses the last LDH value, the LSTM mostly uses early lymphocyte measurements.
- SHAP beat LIME on all four properties. The biggest difference was stability: Spearman 0.97 for SHAP against 0.56 for LIME after adding small noise.
- With only the first 24 hours of data, XGBoost still reaches 0.943 ROC-AUC.
- The most important biomarker changes with the time window: LDH at 24h and for the full stay, lymphocyte % at 48h and 72h.
- The test set only has 13 deaths, so all numbers come with wide confidence intervals (bootstrapped in the notebook).

The full write-up is in the report.

## Running it

```
pip install numpy pandas scikit-learn xgboost shap lime lightgbm seaborn matplotlib scipy torch
```

Put `time_series_375_preprocess_en.xlsx` and `time_series_test_110_preprocess_en.xlsx` in the same folder and run `covid_xai.ipynb` from top to bottom. Plots and tables are saved to `outputs/`. The notebook was run on Google Colab. WindowSHAP is slow, so it runs on 25 test patients by default.

## Use of AI

AI tools were used to help debug and check the code, for example to fix a version conflict between SHAP and XGBoost. All the ideas, the design of the experiments and the execution are our own.
