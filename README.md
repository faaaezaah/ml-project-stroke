# ml-project-stroke
# Predicting Stroke Risk from Patient Health Records

**CS-13410 — Introduction to Machine Learning | Fall 2026 | BSCS**
**The University of Lahore — Faculty of Information Technology, Department of Computer Science & IT**

A semester-long machine learning project that builds, evaluates and interprets models for predicting stroke from routine patient health data. The repository grows part by part as the course progresses.

## Group members

| Name | Student ID |
| --- | --- |
| Faiza Saleem Mian | 70149372 |
| Khadija Shakeel | 70149379 |
| Manahil Shahzad | 70150284 |

## Project overview

Stroke is one of the leading causes of death and long-term disability worldwide. This project asks whether routine patient information (age, hypertension, heart disease, average glucose level, BMI, smoking status and background details) can be used to predict whether a patient has had a stroke, and which factors matter most.

- **Task:** supervised binary classification
- **Target:** `stroke` (1 = had a stroke, 0 = no stroke)
- **Main metrics:** ROC-AUC, PR-AUC, and recall / F1-score of the stroke class. Accuracy is not used, because only about 5% of patients had a stroke.
- **Evaluation rule:** stratified train/test split made before any fitting; the test set stays untouched until final evaluation; all preprocessing lives inside scikit-learn pipelines so nothing leaks from validation or test data.

> This is a university learning project. The models must **not** be used for real medical decisions.

## Dataset

**Stroke Prediction Dataset** by fedesoriano, published on Kaggle:
https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset

- 5,110 patient records, 12 columns
- Features: `gender`, `age`, `hypertension`, `heart_disease`, `ever_married`, `work_type`, `Residence_type`, `avg_glucose_level`, `bmi`, `smoking_status`
- Class balance: 4.9% stroke (249) vs 95.1% no stroke (4,861)
- Data issues: 201 missing BMI values (3.9%), 1,544 "Unknown" smoking entries, outliers in glucose and BMI, one rare gender category, severe class imbalance
- Released on Kaggle for educational use. Credit goes to its author, fedesoriano.

## Project roadmap

| Part | Focus | Status |
| --- | --- | --- |
| **Part 1** | Problem framing, data audit, EDA, leakage-free preprocessing pipeline, class-imbalance handling, sanity-check baseline | Done |
| **Part 2** | Naive Bayes and decision trees, including a hand-computed information-gain split; comparison against the Part 1 baseline | Planned |
| **Part 3** | Ensemble methods (bagging, boosting, random forest), hyperparameter tuning, oversampling comparison inside a pipeline, final evaluation on the held-out test set | Planned |
| **Part 4** | Neural network, clustering and PCA of patient profiles, model interpretability and a fairness discussion | Planned |
| **Part 5** | Final comparison of all models, conclusions, limitations and project report | Planned |

The part titles above follow our project plan and will be updated as each part is submitted.

## Repository contents

| File | Description | Part |
| --- | --- | --- |
| `README.md` | This file | All |
| `healthcare-dataset-stroke-data.csv` | The dataset (optional in Colab: the notebook downloads it automatically if missing) | All |
| `ML_Project_Part1.ipynb` | Problem framing, data audit, EDA, leakage-free pipeline, leakage checks, sanity-check model | 1 |
| `ML_Project_Part1_Proposal.docx` | One-page project proposal | 1 |

New notebooks (`ML_Project_Part2.ipynb`, and so on) are added here as each part is completed.

## Part 1 summary (completed)

- **Data audit:** missing BMI values (informative: patients with missing BMI have a much higher stroke rate), hidden "Unknown" smoking category, no duplicates, identifier dropped.
- **EDA:** seven visualisation sections covering class balance, missing values, age, glucose and BMI, medical history and lifestyle, correlations and outliers.
- **Leakage-free pipeline** (`Pipeline` + `ColumnTransformer`, split before fitting, everything fitted on training data only):
  - median / most-frequent imputation
  - IQR outlier capping at 3x IQR (keeps clinically meaningful high glucose values)
  - standard scaling and one-hot encoding
  - constructed features: `bmi_missing`, `is_senior`, `comorbidity_count` and others
  - feature selection with the ANOVA F-test
  - explicit checks proving the test data never influences the pipeline
- **Class imbalance:** stratified split and cross-validation, class weights.
- **Baseline:** logistic regression with 5-fold stratified cross-validation on the training set only (ROC-AUC about 0.85, recall about 0.81 for the stroke class, precision about 0.14). This is the reference point later parts try to beat.

## How to run (Google Colab)

All notebooks are written for **Google Colab**. Every library they use (pandas, NumPy, scikit-learn, seaborn, matplotlib) is already installed there, so nothing needs to be installed.

**Option A: open directly from GitHub**
1. Go to [colab.research.google.com](https://colab.research.google.com) and choose **File > Open notebook > GitHub**.
2. Paste this repository's URL and pick `ML_Project_Part1.ipynb`.
3. Choose **Runtime > Run all**.

**Option B: upload the notebook**
1. In Colab choose **File > Upload notebook** and select the `.ipynb` file from this repository.
2. Choose **Runtime > Run all**.

You do not need to upload the CSV: if it is not next to the notebook, the notebook downloads a public copy of the dataset automatically. To use your own download from Kaggle instead, set `USE_UPLOAD = True` in the data-loading cell and select `healthcare-dataset-stroke-data.csv` when Colab asks.

*Optional: running locally.* The notebooks also run in Jupyter Notebook or JupyterLab after `pip install numpy pandas matplotlib seaborn scikit-learn jupyter`; keep the CSV in the same folder.

Later parts may need extra packages (for example `imbalanced-learn` or a neural-network library). If so, the notebook will install them with a `!pip install` cell and they will be listed here.

## Acknowledgements

- Dataset: fedesoriano, *Stroke Prediction Dataset*, Kaggle.
- Course: CS-13410 Introduction to Machine Learning, The University of Lahore.
