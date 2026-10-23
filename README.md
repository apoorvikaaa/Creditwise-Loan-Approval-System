# 🏦 CreditWise — Loan Approval Prediction System

**A Machine Learning project that predicts loan approval using applicant financial information, credit history, and demographic attributes.**

## 📌 Overview

CreditWise is a machine learning-based loan approval prediction system designed to assist financial institutions in evaluating loan applications.

Traditional loan evaluation processes can be time-consuming and inconsistent. CreditWise explores how machine learning classification algorithms can analyze historical loan application data and predict whether an applicant's loan should be approved or rejected.

The project applies data preprocessing, exploratory data analysis, feature engineering, and classification algorithms to identify patterns associated with loan approval.

> **Objective:** Build and evaluate machine learning models that predict loan approval using applicant-related features.

## 🎯 Problem Statement

Financial institutions evaluate loan applications using multiple factors, including income, credit scores, existing loans, savings, and debt-to-income ratios.

Manual evaluation can be time-consuming and may lead to inconsistent decisions.

CreditWise addresses this problem by developing a machine learning pipeline that analyzes applicant information and predicts the likelihood of loan approval.

The target variable is `Loan_Approved`:

* `1` — Loan Approved
* `0` — Loan Rejected

The model's prediction is intended to support the evaluation process, not replace final human verification.

## ✨ Features

* **Data Preprocessing:** Identifies and handles missing values using appropriate imputation strategies.
* **Exploratory Data Analysis (EDA):** Examines class distribution, categorical variables, income distributions, outliers, and relationships between financial attributes and loan approval.
* **Categorical Encoding:** Uses Label Encoding and One-Hot Encoding to transform categorical variables into numerical representations.
* **Correlation Analysis:** Uses a correlation matrix and heatmap to explore relationships between numerical features.
* **Feature Scaling:** Applies StandardScaler to standardize input features.
* **Feature Engineering:** Creates squared features for `DTI_Ratio` and `Credit_Score` to explore nonlinear relationships.
* **Model Training:** Trains multiple machine learning classification algorithms.
* **Model Evaluation:** Evaluates predictions using accuracy, precision, recall, F1-score, and confusion matrices.
* **Model Comparison:** Compares Logistic Regression, K-Nearest Neighbors, and Gaussian Naive Bayes.

## 🛠️ Tech Stack

| Technology       | Purpose                                            |
| ---------------- | -------------------------------------------------- |
| Python           | Programming language                               |
| Pandas           | Data manipulation and analysis                     |
| NumPy            | Numerical operations                               |
| Matplotlib       | Data visualization                                 |
| Seaborn          | Statistical data visualization                     |
| Scikit-learn     | Data preprocessing, model training, and evaluation |
| Jupyter Notebook | Interactive development and experimentation        |

## 📂 Project Structure

```text
CreditWise/
│
├── credit_wise.ipynb       # Data analysis and ML implementation
├── loan_approval_data.csv  # Loan application dataset (add this file)
├── README.md               # Project documentation
└── requirements.txt        # Python dependencies (optional but recommended)
```

**Note:** The notebook expects a file named `loan_approval_data.csv` in the working directory. Add the dataset to your repository if you have permission to share it. Otherwise, document how to obtain it.

## 📊 Dataset Description

The dataset represents loan applicants and contains personal, financial, employment, and credit-related attributes.

| Feature              | Description                                     |
| -------------------- | ----------------------------------------------- |
| `Applicant_ID`       | Unique applicant identifier                     |
| `Applicant_Income`   | Monthly applicant income                        |
| `Coapplicant_Income` | Monthly co-applicant income                     |
| `Employment_Status`  | Employment type                                 |
| `Age`                | Applicant age                                   |
| `Marital_Status`     | Marital status                                  |
| `Dependents`         | Number of dependents                            |
| `Credit_Score`       | Credit bureau score                             |
| `Existing_Loans`     | Number of existing loans                        |
| `DTI_Ratio`          | Debt-to-income ratio                            |
| `Savings`            | Applicant savings balance                       |
| `Collateral_Value`   | Value of collateral provided                    |
| `Loan_Amount`        | Requested loan amount                           |
| `Loan_Term`          | Loan duration in months                         |
| `Loan_Purpose`       | Purpose of the loan                             |
| `Property_Area`      | Urban, semi-urban, or rural                     |
| `Education_Level`    | Applicant education level                       |
| `Gender`             | Applicant gender                                |
| `Employer_Category`  | Government, private, or self-employed category  |
| `Loan_Approved`      | Target variable: 1 for approved, 0 for rejected |

The `Applicant_ID` column is removed before model training because it serves as an identifier rather than a meaningful predictive feature.

## ⚙️ Methodology

### 1. Data Loading and Inspection

The dataset is loaded using Pandas.

Initial inspection includes:

* Examining the first few records.
* Checking column data types.
* Identifying missing values.
* Understanding the dataset structure.

### 2. Data Cleaning

Missing values are handled using Scikit-learn's `SimpleImputer`.

* **Numerical features:** Missing values are filled using the mean.
* **Categorical features:** Missing values are filled using the most frequent value.

### 3. Exploratory Data Analysis

EDA is performed to understand the characteristics of the dataset.

The analysis includes:

* Loan approval class distribution.
* Gender and education-level distributions.
* Applicant and co-applicant income distributions.
* Outlier analysis using box plots.
* Credit score distribution across loan approval classes.
* Relationships between loan approval and financial attributes.

### 4. Data Encoding

Categorical variables are converted into numerical representations.

* **Label Encoding:** Applied to `Education_Level` and `Loan_Approved`.
* **One-Hot Encoding:** Applied to employment status, marital status, loan purpose, property area, gender, and employer category.

One-Hot Encoding uses `drop="first"` to avoid retaining every category indicator for each encoded feature.

### 5. Correlation Analysis

A correlation matrix and heatmap are generated to examine relationships between numerical features.

This helps explore which variables are associated with the target variable and identify potential relationships between input features.

### 6. Train-Test Split

The dataset is divided into training and testing sets using an 80:20 split.

* **Training set:** 80% of the data.
* **Testing set:** 20% of the data.
* **Random state:** 42.

The training set is used to fit the models, while the testing set is used to evaluate their predictions.

### 7. Feature Scaling

`StandardScaler` is used to standardize the input features.

The scaler is fitted on the training data and then applied to both the training and testing data.

This ensures that the scaling parameters are learned from the training set rather than the test set.

### 8. Model Training

Three classification algorithms are implemented.

#### Logistic Regression

A classification algorithm that estimates the probability of loan approval based on the input features.

#### K-Nearest Neighbors (KNN)

A classification algorithm that predicts a class using the labels of nearby data points. The implementation uses `n_neighbors=5`.

#### Gaussian Naive Bayes

A probabilistic classification algorithm based on Bayes' theorem that assumes conditional independence between features and models continuous features using Gaussian distributions.

### 9. Model Evaluation

The models are evaluated using the following metrics:

| Metric           | Purpose                                                          |
| ---------------- | ---------------------------------------------------------------- |
| Accuracy         | Measures the proportion of correct predictions                   |
| Precision        | Measures how many predicted positive cases are actually positive |
| Recall           | Measures how many actual positive cases are correctly identified |
| F1-score         | Balances precision and recall                                    |
| Confusion Matrix | Shows correct and incorrect predictions for each class           |

Precision is considered when comparing the models, and the notebook identifies Gaussian Naive Bayes as the preferred model based on precision.

Actual evaluation scores should be added to this README after running the notebook.

### 10. Feature Engineering

The project explores whether squared transformations of selected financial features improve model performance.

Two additional features are created:

* `DTI_ratio_sq` — Squared debt-to-income ratio.
* `Credit_Score_sq` — Squared credit score.

The transformed dataset is then used to retrain and evaluate the classification models.

The effectiveness of these transformations should be determined by comparing the evaluation metrics before and after feature engineering.

## 🚀 Getting Started

Follow these steps to run the project locally.

### Prerequisites

Install the following:

* Python 3.10 or later
* Jupyter Notebook
* pip

### 1. Clone the Repository

Replace `<your-username>` with your GitHub username.

```bash
git clone https://github.com/<your-username>/CreditWise.git
```

Navigate to the project directory:

```bash
cd CreditWise
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Alternatively, create a `requirements.txt` file containing:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

### 4. Add the Dataset

Place `loan_approval_data.csv` in the project directory alongside the notebook.

Ensure the dataset contains the columns expected by the notebook, including `Loan_Approved`.

### 5. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open `credit_wise.ipynb` and execute the cells in order.

**Important:** The notebook must be able to locate the dataset using the path specified in `pd.read_csv()`.

## 📈 Results

The project compares three machine learning models:

* Logistic Regression
* K-Nearest Neighbors
* Gaussian Naive Bayes

The models are evaluated using accuracy, precision, recall, F1-score, and confusion matrices.

Gaussian Naive Bayes is identified in the notebook as the preferred model based on precision. The notebook also explores feature engineering by adding squared financial features.

**Quantitative results:** Add the actual metric values obtained from your notebook before publishing the final version of this README. No accuracy or precision values are assumed here.

## 🔮 Future Improvements

Potential improvements include:

* Compare additional classification algorithms, such as Decision Trees, Random Forest, and Gradient Boosting.
* Use cross-validation and hyperparameter tuning to improve model selection.
* Compare model performance before and after feature engineering.
* Evaluate class imbalance and use appropriate techniques if necessary.
* Use a Scikit-learn `Pipeline` to organize preprocessing and model training.
* Fit imputers and encoders only on training data to prevent data leakage.
* Evaluate fairness across demographic groups and investigate potential bias.
* Add explainability using SHAP or other model interpretation techniques.
* Develop a web interface using Flask, FastAPI, or Streamlit.
* Deploy the selected model as an API for loan prediction.
* Add automated tests and model monitoring.

## ⚠️ Limitations and Responsible Use

This project is an educational machine learning implementation.

Loan approval is a high-impact financial decision. Predictions should not be treated as definitive lending decisions without appropriate validation, human oversight, and compliance with applicable laws and regulations.

Model performance depends on the quality, representativeness, and reliability of the training data.

Features such as gender and other demographic attributes can introduce fairness concerns. Their use requires careful legal, ethical, and statistical evaluation.

The model's predictions should not be interpreted as proof that an applicant is creditworthy or uncreditworthy.
