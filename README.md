# CodeAlpha Credit Scoring Model

Machine Learning Internship Task 1 at CodeAlpha.

## Objective
Predict an individual's creditworthiness (good / bad credit) using past financial data.

## Dataset
German Credit Data (UCI Machine Learning Repository), 1000 records with features like credit amount, duration, age, employment, savings and payment history.

## Approach
- Feature engineering: `credit_per_month`, `age_group`
- Preprocessing: StandardScaler (numeric), OneHotEncoder (categorical)
- Models: Logistic Regression, Decision Tree, Random Forest
- Evaluation: Precision, Recall, F1-Score, ROC-AUC

## Results
| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| Logistic Regression | 0.784 | 0.829 | 0.806 | 0.763 |
| Decision Tree | 0.766 | 0.771 | 0.769 | 0.679 |
| Random Forest | 0.804 | 0.907 | 0.852 | 0.785 |
