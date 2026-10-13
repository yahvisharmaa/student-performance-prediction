# Student Performance Prediction and Analysis

## Project overview
This project demonstrates a small predictive-analytics workflow for estimating student exam scores and predicting pass/fail outcomes from study habits and prior academic indicators.

**Important data note:** the included CSV is a reproducible *synthetic/simulated dataset* created for this educational project. It is not real student data and should not be presented as a real-world survey.

## Research questions
1. Can exam scores be estimated from study hours, attendance, previous score, sleep, and assignment completion?
2. Can a model classify whether a student is likely to pass (score >= 40)?
3. How do baseline models compare on held-out test data?

## Methods
- Data inspection and basic exploratory analysis
- Train/test split (80/20, fixed random seed)
- Regression: Linear Regression and Random Forest Regressor
- Classification: Logistic Regression and Random Forest Classifier
- Regression metrics: MAE, RMSE, R²
- Classification metrics: accuracy, precision, recall, F1-score
- Visualizations: actual vs predicted scores, feature importance, confusion matrix

## Repository structure
```text
student-performance-prediction/
├── README.md
├── requirements.txt
├── data/
│   └── student_performance.csv
├── notebooks/
│   └── student_performance_analysis.ipynb
├── src/
│   └── train_model.py
└── outputs/                 # created when the script runs
```

## Run the project
Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
python src/train_model.py
```

The script prints model evaluation metrics and saves plots in `outputs/`.

Alternatively, open `notebooks/student_performance_analysis.ipynb` in Google Colab or Jupyter and run the cells from top to bottom.

## Key limitations
- Data is simulated, so metrics illustrate the workflow rather than real-world predictive performance.
- Relationships in simulated data are simplified and may not reflect real student populations.
- Predictions are estimates, not judgments about a student's ability; real deployment would require representative, consented data and fairness checks.

## Conclusion
This project shows how predictive analytics can combine regression and classification to support an educational question. The better model should be chosen using held-out evaluation metrics rather than training performance alone.
