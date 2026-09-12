# CreditWise - Loan Prediction System

CreditWise is a Machine Learning based Loan Approval Prediction System that predicts whether a loan application is likely to be approved based on applicant and financial information.

## Project Overview

The project uses applicant-related financial and demographic features to build a machine learning model for loan approval prediction.

The complete machine learning workflow is implemented in a Jupyter Notebook, including:

- Data loading
- Data preprocessing
- Exploratory Data Analysis
- Feature engineering
- Feature scaling
- Model training
- Model evaluation
- Loan approval prediction

## Dataset

The dataset contains applicant information such as:

- Applicant Income
- Coapplicant Income
- Employment Status
- Age
- Dependents
- Credit Score
- Existing Loans
- DTI Ratio
- Savings
- Collateral Value
- Loan Amount
- Loan Term
- Loan Purpose
- Property Area
- Education Level
- Gender
- Employer Category

Target variable:

`Loan_Approved`

## Machine Learning Models

The project experiments with:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

Gaussian Naive Bayes was selected as the final model in the notebook.

## Model Performance

The final Gaussian Naive Bayes model achieved approximately:

- Accuracy: 86.5%
- Precision: 80.36%
- Recall: 73.77%
- F1 Score: 76.92%

## Feature Engineering

Additional features were created using:

- Squared DTI Ratio
- Squared Credit Score

These features were used during the final model training process.

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Joblib

## Project Structure

```text
CreditWise/
│
├── CreditWise_Loan_Prediction.ipynb
├── loan_approval_data.csv
├── requirements.txt
├── README.md
│
└── models/
    ├── loan_model.pkl
    └── scaler.pkl