# Machine Learning Assignment 2 - Classification Models and Streamlit Deployment

## a. Problem Statement

The objective of this assignment is to build and compare multiple supervised classification models on one public classification dataset. The trained models are exposed through an interactive Streamlit web application where users can upload test data, select a model, and view evaluation results.

## b. Dataset Description

Dataset selected: Breast Cancer Wisconsin Diagnostic dataset from the UCI Machine Learning Repository.

- Task type: Binary classification
- Number of instances: 569
- Number of input features: 30
- Target classes: malignant and benign
- Minimum assignment requirement satisfied: more than 500 instances and more than 12 features
- Source: UCI Machine Learning Repository / sklearn built-in dataset loader

The target column used in this project is `target`.

- 0 = malignant
- 1 = benign

## c. GitHub Repository Link

https://github.com/syayounus/ML_assignment2.git

## d. Live Streamlit App Link

https://mlassignment2-deofcsapppfqtpv6lctsvjp.streamlit.app/

## e. Models Used and Evaluation Metrics

All models were trained and evaluated on the same train-test split with `random_state=42`.

| ML Model Name            |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-------------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression      |     0.986  | 0.9977 |      0.9889 |   0.9889 | 0.9889 | 0.97   |
| Decision Tree            |     0.9371 | 0.9186 |      0.9551 |   0.9444 | 0.9497 | 0.8657 |
| kNN                      |     0.979  | 0.9845 |      0.9677 |   1      | 0.9836 | 0.9555 |
| Naive Bayes              |     0.9371 | 0.9893 |      0.9263 |   0.9778 | 0.9514 | 0.865  |
| Random Forest (Ensemble) |     0.958  | 0.9949 |      0.9565 |   0.9778 | 0.967  | 0.9098 |

## f. Observations on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Performed very strongly because the dataset has mostly numerical features with clear linear separation after scaling. |
| Decision Tree | Good and easy to interpret, but slightly more sensitive to train/test split and can overfit if depth is not controlled. |
| kNN | Performed well after feature scaling because distance-based comparison works better when all features are on comparable scales. |
| Naive Bayes | Fast and simple baseline. It performed reasonably but can be weaker when feature independence assumptions are not fully true. |
| Random Forest (Ensemble) | Strong overall performance because it combines many decision trees and reduces single-tree overfitting. |
| Overall Winner for this dataset | Logistic Regression, based on the strongest combination of F1, MCC, and AUC. |

## g. Streamlit App Features

The app provides the following required features:

- CSV test data upload option
- Model selection dropdown
- Evaluation metrics display
- Confusion matrix display
- Classification report display
- Prediction output preview

## h. Repository Structure

```text
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- train_data.csv
|-- model_metrics.csv
|-- model/
|   |-- train_models.py
|   |-- logistic_regression.pkl
|   |-- decision_tree.pkl
|   |-- knn.pkl
|   |-- naive_bayes.pkl
|   |-- random_forest_ensemble.pkl
|   |-- feature_columns.json
|   |-- target_names.json
|   |-- evaluation_reports.json
```

## i. How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload `test_data.csv`, select any model from the dropdown, and review the metrics.

## j. Deployment Steps

1. Upload this project folder to a new GitHub repository.
2. Go to Streamlit Community Cloud.
3. Click New App.
4. Select the GitHub repository and branch.
5. Set the main file path as `app.py`.
6. Click Deploy.
7. Copy the deployed app URL and replace the placeholder in this README and in the final PDF.

## k. BITS Virtual Lab Screenshot

Run the project once on BITS Virtual Lab and take one screenshot showing the assignment execution. Add that screenshot to the final submitted PDF.
