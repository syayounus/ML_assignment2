import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
feature_columns = list(data.feature_names)
X_train, X_test, y_train, y_test = train_test_split(df[feature_columns], df["target"], test_size=0.25, random_state=42, stratify=df["target"])

models = {
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000, random_state=42))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "kNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]),
    "Naive Bayes": GaussianNB(),
    "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=200, random_state=42),
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_test, prob),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1": f1_score(y_test, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred),
    })
    file_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "") + ".pkl"
    joblib.dump(model, MODEL_DIR / file_name)

pd.DataFrame(rows).to_csv(BASE_DIR / "model_metrics.csv", index=False)
X_test.assign(target=y_test.values).to_csv(BASE_DIR / "test_data.csv", index=False)
with open(MODEL_DIR / "feature_columns.json", "w", encoding="utf-8") as file:
    json.dump(feature_columns, file, indent=2)
with open(MODEL_DIR / "target_names.json", "w", encoding="utf-8") as file:
    json.dump(list(data.target_names), file, indent=2)
print("Training completed. Model files and metrics are saved.")
