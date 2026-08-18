import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

st.set_page_config(page_title="ML Assignment 2 Classification App", layout="wide")

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
TARGET_COL = "target"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "kNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest (Ensemble)": "random_forest_ensemble.pkl",
}

@st.cache_resource
def load_model(model_name):
    return joblib.load(MODEL_DIR / MODEL_FILES[model_name])

@st.cache_data
def load_feature_columns():
    with open(MODEL_DIR / "feature_columns.json", "r", encoding="utf-8") as file:
        return json.load(file)

@st.cache_data
def load_target_names():
    with open(MODEL_DIR / "target_names.json", "r", encoding="utf-8") as file:
        return json.load(file)


def calculate_metrics(y_true, y_pred, y_probability):
    auc_value = roc_auc_score(y_true, y_probability) if len(np.unique(y_true)) == 2 else np.nan
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc_value,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, zero_division=0),
        "MCC Score": matthews_corrcoef(y_true, y_pred),
    }

st.title("Machine Learning Assignment 2 - Classification Model Dashboard")
st.write("This Streamlit app evaluates multiple classification models on uploaded breast cancer test data.")

with st.sidebar:
    st.header("Controls")
    selected_model = st.selectbox("Select ML model", list(MODEL_FILES.keys()))
    uploaded_file = st.file_uploader("Upload test CSV file", type=["csv"])
    st.caption("Use the included test_data.csv file for evaluation.")

feature_columns = load_feature_columns()
target_names = load_target_names()
model = load_model(selected_model)

if uploaded_file is None:
    st.info("Please upload test_data.csv to view model evaluation results.")
    st.stop()

data = pd.read_csv(uploaded_file)
st.subheader("Uploaded Test Data Preview")
st.dataframe(data.head(), use_container_width=True)

missing_features = [col for col in feature_columns if col not in data.columns]
if missing_features:
    st.error(f"Missing required feature columns: {missing_features}")
    st.stop()
if TARGET_COL not in data.columns:
    st.error("The uploaded CSV must include a target column named 'target'.")
    st.stop()

X = data[feature_columns]
y_true = data[TARGET_COL]
y_pred = model.predict(X)
y_probability = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else y_pred

metrics = calculate_metrics(y_true, y_pred, y_probability)

st.subheader(f"Evaluation Metrics - {selected_model}")
metric_cols = st.columns(6)
for col, (metric_name, metric_value) in zip(metric_cols, metrics.items()):
    col.metric(metric_name, f"{metric_value:.4f}" if not np.isnan(metric_value) else "NA")

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)

st.subheader("Classification Report")
report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)

st.subheader("Prediction Output")
output = data.copy()
output["predicted_target"] = y_pred
output["predicted_label"] = [target_names[int(value)] for value in y_pred]
st.dataframe(output.head(50), use_container_width=True)
