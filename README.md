# ISRO-BurnIn-AI

AI-driven anomaly detection for electronic component burn-in screening.

## Problem Statement

Electronic components used in high-reliability systems undergo burn-in and screening tests to identify early failures.

Traditional screening mainly uses fixed pass/fail limits. However, some components may remain within these limits while showing abnormal degradation over time.

This project uses data-driven techniques to identify such abnormal component behavior and assign a risk level.

## Proposed Solution

The system uses two complementary modules:

### Module A — Lot-Based Statistical Anomaly Detection

Module A compares the 168-hour component value with the statistical behavior of its production lot.

- Calculates lot-wise mean and standard deviation
- Computes a Z-score for each component
- Flags components whose deviation crosses the selected threshold
- Final threshold used: Z-score > 1.0
- Helps identify components showing unusual behavior within their lot

### Module B — Early-Stage Machine Learning Prediction

Module B uses early burn-in measurements to predict whether a component is likely to be anomalous.

Input features:

- Value at 0h
- Value at 24h
- Early Drift = Value_24h − Value_0h

Machine learning model:

- Random Forest Classifier
- 80% training data
- 20% testing data
- Class balancing enabled

The model also produces an anomaly risk score:

- Below 30% → Low Risk
- 30%–69% → Warning
- 70% and above → High Risk

## Dataset

A synthetic dataset was created specifically for this project because no suitable public dataset was found for burn-in component screening with time-series measurements.

Dataset details:

- 1000 electronic components
- 10 production lots
- Measurements at 0h, 24h, 96h and 168h
- Component behavior categories:
  - Normal
  - Gradual Degradation
  - Sudden Degradation
  - Latent Defect

The dataset is available in:

`data/synthetic_burnin_dataset.csv`

## Results

### Module A

Lot-based statistical anomaly detection achieved:

- Accuracy: 89%
- Anomaly Precision: 100%
- Anomaly Recall: 63%
- Anomaly F1-score: 77%

### Module B

Random Forest early-stage prediction achieved:

- Accuracy: 89%
- Anomaly Precision: 93%
- Anomaly Recall: 67%
- Anomaly F1-score: 78%

The results show that early measurement changes can provide useful signals for identifying potentially anomalous components.

## Project Structure

```text
ISRO-BurnIn-AI/
│
├── data/
│   └── synthetic_burnin_dataset.csv
│
├── module_a/
│   └── module_a.py
│
├── module_b/
│   └── module_b.py
│
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib

## References

1. NASA Small Spacecraft Systems Virtual Institute — Burn-In Testing of Electronics  
   https://s3vi.ndc.nasa.gov/ssri-kb/topics/47/

2. Scikit-learn — Random Forest Classifier  
   https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

3. UCI Machine Learning Repository — SECOM Dataset  
   https://archive.ics.uci.edu/dataset/179/secom

## GitHub Repository

This repository contains the dataset, anomaly detection modules, machine learning model code and project documentation.
