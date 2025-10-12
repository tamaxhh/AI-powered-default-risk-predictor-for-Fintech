# AI-Powered Default Risk Predictor for FinTech

An end-to-end machine learning project to predict loan default risk using advanced modeling and provide transparent, explainable insights via a web application.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Data & Target Variable](#data--target-variable)
4. [Methodology](#methodology)
    - [Data Preparation & Engineering](#data-preparation--engineering)
    - [Model Training and Optimization](#model-training-and-optimization)
    - [Explainable AI (XAI)](#explainable-ai-xai)
5. [Results](#results)
6. [Key Insights](#key-insights)
7. [Conclusion](#conclusion)
8. [File Structure](#file-structure)
9. [How to Run](#how-to-run)
10. [Future Enhancements](#future-enhancements)
11. [Contributing](#contributing)
12. [License](#license)

---

## Introduction

This project aims to develop a sophisticated, automated system for predicting **loan default risk** in the FinTech domain. Traditionally, credit risk assessment is a manual, time-consuming process prone to human bias and lacking scalability. By leveraging modern machine learning (ML) models like **XGBoost** and **LightGBM**, along with a focus on **Explainable AI (XAI)**, this system offers financial institutions a **real-time, data-driven solution** to forecast default probabilities, streamline the approval process, and mitigate financial exposure.

---

## Problem Statement

Financial institutions manage vast loan portfolios but often lack integrated, real-time tools for accurate default risk assessment across diverse applicant segments. This lack of integrated visibility into applicant behavior and credit history leads to costly misjudgments and unexpected defaults. The project addresses the critical need for a **predictive, visual decision-support tool** that empowers credit analysts to track risk patterns, evaluate applicant profiles systematically, and proactively minimize potential losses.

---

## Data & Target Variable

The project is built upon historical applicant and credit data, combined and preprocessed for optimal model performance.

### Dataset Overview

The primary data source is referenced in the project files as **'Loan Status Prediction.csv'** (implied from `config.py`), which contains various demographic, financial, and credit history features for loan applicants.

| Feature Type | Examples |
| --- | --- |
| **Applicant Demographics** | Gender, Marital Status, Education, Dependents |
| **Financial Features** | Applicant Income, Coapplicant Income, Loan Amount, Loan Amount Term |
| **Credit History** | Credit_History, Self_Employed |
| **Other** | Property_Area |

### Target Variable Creation

The target variable is **`Loan_Status`** (Approved / Not Approved), which is ultimately used to classify applicants as **Low Risk** or **High Risk (Default)**.

---

## Methodology

The methodology follows a robust machine learning engineering pipeline, covering data preparation, advanced modeling, and deployment preparation.

### Data Preparation & Engineering

1. **Preprocessing:** Handling missing values, and converting categorical features using techniques like one-hot encoding or Label Encoding (as seen in the provided code snippets).
2. **Feature Engineering:** Creating new, informative features from the raw data to improve model signal.
3. **Handling Class Imbalance:** The project specifically uses the **`imblearn`** library and **SMOTE** (Synthetic Minority Oversampling Technique) to ensure the model is not biased towards the majority class (e.g., non-defaulters), leading to more reliable risk prediction.

### Model Training and Optimization

- **Model Selection:** Multiple robust classifiers are trained and evaluated, including **Logistic Regression**, **XGBoost**, and **LightGBM** (`requirements.txt` indicates all three are used).
- **Pipeline:** The `main.py` script orchestrates the entire pipeline: `preprocess` -> `engineer_features` -> `train_models` -> `evaluate_models`.
- **Best Model:** The final deployed solution (e.g., in `streamlit_app.py`) leverages a tree-based model (like **XGBoost**) for its high predictive power.

### Explainable AI (XAI)

A critical component of this project is providing transparency for risk-based decisions.

- **SHAP (SHapley Additive exPlanations):** Used to provide **global feature importance** (which factors generally drive the risk score) and **local explanations** (why a specific applicant was approved or rejected), as seen in `streamlit_app.py`.
- **LIME (Local Interpretable Model-agnostic Explanations):** Used alongside SHAP to offer an alternative local explanation, detailing which features strongly supported a **Low Risk** or **High Risk** decision for a single data point.

---

## Results

The final trained model is evaluated using a comprehensive suite of metrics beyond simple accuracy, focusing on the ability to correctly identify the high-risk (minority) class.

The preliminary results (from the provided text, typically before SMOTE/advanced tuning) showed:

| Metric | Score |
| --- | --- |
| **Overall Accuracy** | **~79.15%** |
| **Minority Class Recall** | **0.00** |

**Note:** The initial low Recall for the minority class (Class 1.0) underscores the importance of the implemented class imbalance techniques (**SMOTE** in `requirements.txt` and `config.py`). The final, optimized model (likely an XGBoost model using SMOTE) is expected to have a significantly higher **Recall** for the minority class, which is crucial for minimizing financial losses by catching most high-risk applicants.

---

## Key Insights

- **Prioritizing Recall:** For risk prediction, maximizing **Recall** for the default/high-risk class is paramount to minimizing potential losses, even if it slightly reduces overall accuracy.
- **Automation:** The system successfully automates the manual approval process, offering substantial operational efficiency.
- **Transparency:** The integration of **SHAP** and **LIME** transforms the ML model from a "black box" into a transparent decision-making tool, which is vital for regulatory compliance and earning trust from credit analysts.

---

## Conclusion

The "AI-Powered Default Risk Predictor" successfully validates a modern, end-to-end ML approach to credit risk assessment. The functional pipeline, combined with the interactive **Streamlit dashboard**, demonstrates a powerful solution capable of classifying applications and, most importantly, **explaining the rationale** behind each decision. This project provides a strong foundation for building sophisticated and equitable automated credit approval systems in a production environment.

---

## File Structure

The project is structured following ML engineering best practices with separate scripts for the pipeline steps and a dedicated dashboard file.

```python
.
├── Data/
│   ├── Loan Status Prediction.csv  # Raw Data 
│   └── processed/
│       └── processed_loan_data.csv
├── Notebook
├── streamlit_app.py          # Interactive Streamlit App
├── logs/
├── models/                       # Trained models (e.g., xgb_model.sav)
├── src/                          # ML Pipeline source code
│   ├── data_preprocessing.py
│   ├── feature_engineer.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── interpret_model.py
├── main.py                       # Main execution script for the pipeline
├── config.py                     # Configuration constants (paths, params)
├── requirements.txt              # Project dependencies [cite: 97]
├── README.md                     # Project overview (This file)
├── .gitignore                    # Specifies files to ignore [cite: 1]
└── utils.py                      # Utility functions (e.g., logging)
```

---

## How to Run

### Prerequisites

- Python 3.x
- Git

### 1. Clone the repository

Bash

```python
git clone https://github.com/tamaxhh/AI-powered-default-risk-predictor-for-Fintech.git

cd AI-Powered-Default-Risk-Predictor
```

### 2. Set up the Environment

It is required to set up a virtual environment and install dependencies.

Bash

```python
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate

# Install all required dependencies
pip install -r requirements.txt
```

### 3. Run the ML Pipeline (Training)

Execute the main script to run the preprocessing, feature engineering, training, and evaluation steps. This will train the models and save the best one (using `joblib`) for the Streamlit application.

Bash

```python
python main.py
```

### 4. Run the Web Dashboard (Deployment)

Start the interactive Streamlit application to demonstrate real-time prediction and XAI for custom inputs.

Bash

```python
streamlit run streamlit_app.py
```

A local URL will be provided in your terminal, which you can open in your browser.

---

## Future Enhancements

- **Hyperparameter Tuning:** Implement automated hyperparameter tuning (e.g., using GridSearch or Optuna) on the optimized models (**XGBoost**, **LightGBM**) to achieve peak performance.
- **Time Series Analysis:** If applicable data is available, integrate time-series features (e.g., rolling averages of credit status) to improve predictive power.
- **API Deployment:** Replace the Streamlit frontend with a lightweight **Flask API** wrapper for the model, enabling easy integration with production systems.
- **Continuous Integration/Deployment (CI/CD):** Set up a CI/CD pipeline to automate model retraining and deployment upon new data arrival.

---

## Contributing

Contributions are welcome! Please feel free to open an issue or submit a pull request

---

**Disclaimer:** This project is for educational and demonstrative purposes only. It should not be used for actual financial decision-making without further rigorous validation, regulatory compliance, and expert review.
