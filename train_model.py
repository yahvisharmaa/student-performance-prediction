"""
Student Performance Prediction and Analysis
Run from the repository root: python src/train_model.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    ConfusionMatrixDisplay
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "student_performance.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
features = [
    "study_hours_per_day", "attendance_percent", "previous_exam_score",
    "sleep_hours", "assignment_completion_percent"
]
X = df[features]
y_reg = df["final_exam_score"]
y_cls = df["passed"]

X_train, X_test, yr_train, yr_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)
# Use the same split seed for classification.
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

regressors = {
    "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=150, random_state=42, min_samples_leaf=3
    ),
}
reg_rows = []
predictions = {}
for name, model in regressors.items():
    model.fit(X_train, yr_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    reg_rows.append({
        "Model": name,
        "MAE": mean_absolute_error(yr_test, pred),
        "RMSE": np.sqrt(mean_squared_error(yr_test, pred)),
        "R2": r2_score(yr_test, pred),
    })
reg_results = pd.DataFrame(reg_rows).sort_values("RMSE")
print("\\nREGRESSION RESULTS (lower MAE/RMSE and higher R² are better)")
print(reg_results.round(3).to_string(index=False))
reg_results.to_csv(OUT / "regression_metrics.csv", index=False)

plt.figure(figsize=(7, 5))
best_reg = reg_results.iloc[0]["Model"]
plt.scatter(yr_test, predictions[best_reg], alpha=0.75)
plt.plot([0, 100], [0, 100], linestyle="--")
plt.xlabel("Actual exam score")
plt.ylabel("Predicted exam score")
plt.title(f"Actual vs Predicted Scores — {best_reg}")
plt.tight_layout()
plt.savefig(OUT / "actual_vs_predicted.png", dpi=160)
plt.close()

classifiers = {
    "Logistic Regression": make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)
    ),
    "Random Forest Classifier": RandomForestClassifier(
        n_estimators=150, random_state=42, min_samples_leaf=3
    ),
}
cls_rows = []
cls_models = {}
for name, model in classifiers.items():
    model.fit(Xc_train, yc_train)
    pred = model.predict(Xc_test)
    cls_models[name] = model
    cls_rows.append({
        "Model": name,
        "Accuracy": accuracy_score(yc_test, pred),
        "Precision": precision_score(yc_test, pred, zero_division=0),
        "Recall": recall_score(yc_test, pred, zero_division=0),
        "F1": f1_score(yc_test, pred, zero_division=0),
    })
cls_results = pd.DataFrame(cls_rows).sort_values("F1", ascending=False)
print("\\nCLASSIFICATION RESULTS (higher scores are better)")
print(cls_results.round(3).to_string(index=False))
cls_results.to_csv(OUT / "classification_metrics.csv", index=False)

best_cls_name = cls_results.iloc[0]["Model"]
best_cls = cls_models[best_cls_name]
ConfusionMatrixDisplay.from_estimator(best_cls, Xc_test, yc_test)
plt.title(f"Confusion Matrix — {best_cls_name}")
plt.tight_layout()
plt.savefig(OUT / "confusion_matrix.png", dpi=160)
plt.close()

# Feature importance from the random forest models
rf_reg = regressors["Random Forest Regressor"]
importance = pd.Series(rf_reg.feature_importances_, index=features).sort_values()
plt.figure(figsize=(7, 5))
importance.plot(kind="barh")
plt.xlabel("Feature importance")
plt.title("Random Forest Regressor — Feature Importance")
plt.tight_layout()
plt.savefig(OUT / "feature_importance.png", dpi=160)
plt.close()
importance.rename("importance").to_csv(OUT / "feature_importance.csv")

print(f"\\nSaved metrics and plots to: {OUT}")
print("Data reminder: the included dataset is simulated, not real student records.")
