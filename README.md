# SHAP_Analysis_of_Credit_Risk_Prediction
A machine learning project that predicts credit loan default risk using XGBoost and explains model decisions using SHAP for full transparency and interpretability.

# 🏦 Credit Risk Prediction using XGBoost + SHAP (Interpretable Machine Learning)

## 📌 Project Overview
This project aims to build a **credit loan default prediction model** and make the results **fully explainable** using **SHAP (SHapley Additive exPlanations)**.  
The model predicts whether a customer will **default (1) or not default (0)** based on demographic, financial, and behavioral features.

Traditional ML models work as black boxes—but this project focuses equally on:
✔️ **Prediction accuracy**  
✔️ **Interpretability**  
✔️ **Fair and transparent decision-making**

---

## 🚀 Key Features of This Project
- **Machine Learning Pipeline**: Preprocessing → Encoding → SMOTE → Scaling → Training  
- **Model Used**: XGBoost Classifier  
- **Hyperparameter Tuning**: GridSearchCV  
- **Model Evaluation**: Classification report, ROC-AUC  
- **Explainability**:  
  - SHAP summary plot  
  - SHAP bar plot (global importance)  
  - SHAP local explanation for specific high-risk customers  
- **Imbalanced Data Handling**: SMOTE oversampling  

---


---

## 🛠️ Technologies and Libraries Used
- Python  
- Pandas, NumPy  
- Matplotlib, Seaborn  
- Scikit-learn  
- SMOTE (imbalanced-learn)  
- XGBoost  
- SHAP  

---

# 🧠 Project Workflow (Step-by-Step Explanation)

## **1️⃣ Importing Required Libraries**
All essential libraries for ML modeling, tuning, evaluation, and interpretability are imported.

## **2️⃣ Loading the Dataset**
```python
df = pd.read_csv("credit_risk_dataset.csv")
```
- Displays dataset shape

- Shows first 5 rows

- Helps check data quality

## **3️⃣ Data Preprocessing**

✔️ Checks missing values
✔️ Fills missing values using backward fill
✔️ Encodes all categorical columns using LabelEncoder
```python
df.fillna(method='bfill', inplace=True)
```

## **4️⃣ Splitting Features and Target**
```python
X = df.drop("loan_status", axis=1)
y = df["loan_status"]
```

loan_status is the target variable (0 or 1)

## **5️⃣ Train–Test Split**

Stratified split ensures correct class proportion.

```python
train_test_split(... stratify=y)
```

## **6️⃣ Handling Imbalance Using SMOTE**

Loan default datasets are mostly imbalanced.
SMOTE synthesizes new minority samples.

```python
smote = SMOTE()
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

## **7️⃣ Feature Scaling**

Scaling is optional for XGBoost but improves interpretability in SHAP visualizations.

## **8️⃣ Training XGBoost with Hyperparameter Tuning**

```python
params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1]
}
```

GridSearchCV performs:

- Multiple training runs

- Cross-validation

- Selection of best parameters

Final model:

```python
best_model = grid.best_estimator_
```

## **9️⃣ Model Evaluation**

- Classification Report

- ROC-AUC Score

- Confusion Matrix
```python
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
```
## 📊 Model Insights

### ⭐ Key Features Influencing Default:
- Income level  
- Credit history length  
- Employment stability  
- Debt-to-income ratio  
- Interest rate  

### ⭐ Why SHAP is Important
SHAP helps explain:
- Why each borrower was classified as **high risk**  
- Which features increase the risk score  
- Whether the model is fair and transparent  

---

## 📝 Conclusion

This project demonstrates a complete **Explainable Machine Learning workflow** using:
- **XGBoost** for strong predictive performance  
- **SMOTE** to fix imbalance  
- **SHAP** for interpretability  

It provides both **global** and **local** model explanations, making it suitable for finance, banking, and regulatory use cases.

---

## Expected Deliverables

1 The complete, runnable Python code for data preparation, modeling, tuning, and SHAP analysis (provided as plain text).
### **1️⃣ Complete Runnable Python Code**
A full end-to-end Python script is included, covering:
- Data loading  
- Data cleaning and preprocessing  
- Label encoding and feature scaling  
- SMOTE for class balancing  
- XGBoost model training  
- GridSearchCV hyperparameter tuning  
- Evaluation (AUC, Precision, Recall, F1-score)  
- SHAP global + local interpretability  

The entire code is provided inside credit_risk_shap_full.py
---
2 A text-based report detailing the chosen model architecture, hyperparameter tuning results, and performance metrics (AUC, Precision, Recall).
### **2️⃣ Text-Based Model Report**
A detailed written summary is included in this README describing:
- The chosen model architecture (XGBoost with specific parameters)  
- Best hyperparameters found by GridSearchCV  
- Model performance metrics:  
  - **ROC-AUC Score**  
  - **Precision**  
  - **Recall**  
  - **F1-score**  
  - **Classification Report**  

This report explains why XGBoost was chosen, how hyperparameters improved performance, and what the metrics tell us about model quality.
view full report in reports/auto_report.txt
---
3 A textual analysis comparing global feature importance derived from standard model metrics versus SHAP values
### **3️⃣ Comparison of Global Feature Importance** 
A textual analysis is provided comparing:
- **Native XGBoost feature importance values**  
- **SHAP global feature importance (mean absolute SHAP values)**  

Key comparison points include:
- Which features XGBoost considers important  
- How SHAP gives deeper insight into positive/negative impacts  
- Why SHAP is preferred for financial explainability  
view full answer in Textual_analysis.txt
---

4 Textual descriptions interpreting the local SHAP explanations (force plots) for the three selected high-risk cases, including specific variable contributions.
### **4️⃣ Local SHAP Explanation Descriptions**
Below are three example textual interpretations you can use as templates. They are deliberately concrete and refer to common credit features. After running the script, replace the feature names and SHAP numbers with ones from reports/high_risk_shap_contributions.csv and reports/local_shap_summaries.txt.
Three high-risk test cases are selected, and for each:
- The top contributing features  
- Whether each feature increased or decreased default probability  
- A human-readable explanation of the force plot  

This provides transparency for individual predictions—critical in finance and loan decision auditing.
Below are three example textual interpretations you can use as templates. They are deliberately concrete and refer to common credit features. After running the script, replace the feature names and SHAP numbers with ones from reports/high_risk_shap_contributions.csv and reports/local_shap_summaries.txt.

### Example Local SHAP Interpretation — High-Risk Case 1

Test index: 1345 (example)
Predicted probability of default: 0.92 (92%)
True label: 1 (default) — if known

**Top positive contributors (pushing model → default):**

- past_default (SHAP +0.75): Applicant has previous defaults — the largest single risk driver.

- debt_to_income_ratio (SHAP +0.42): High DTI increases stress on repayment ability.

- loan_amount (SHAP +0.31): Requested loan size is much larger than income/typical loans, raising risk.

- interest_rate (SHAP +0.12): Higher interest rate increased repayment burden.

- num_recent_inquiries (SHAP +0.08): Multiple recent credit inquiries indicate credit-seeking behavior.

**Top negative contributors (pushing model ← non-default):**

- employment_length (SHAP -0.10): Longer employment reduced risk slightly.

- annual_income (SHAP -0.05): Income level provided a small offset to risk.

**Interpretation & action:**
The model's prediction is primarily driven by a history of prior default and an elevated debt-to-income ratio. For underwriting, this suggests manual review focusing on repayment context (was prior default due to one-time event?) and verification of income/debt figures. If the customer can demonstrate mitigating circumstances or provide collateral, the risk may be mitigated.

### Example Local SHAP Interpretation — High-Risk Case 2

Test index: 2876
Predicted probability of default: 0.86 (86%)
True label: 0 (non-default) — if known

**Top positive contributors:**

- credit_history_length (SHAP +0.50): Very short history — model treats this as higher risk.

- loan_purpose = 'debt_consolidation' (SHAP +0.22): Some purposes are associated with higher default rates historically.

- loan_amount_to_income (SHAP +0.15): Loan relative to income is high.

**Top negative contributors:**

- no_of_open_accounts (SHAP -0.20): Multiple active accounts with good standing reduce risk.

- home_ownership (SHAP -0.10): Owning a home reduces risk slightly.

**Interpretation & action:**
Here, the short credit history and loan purpose mainly push the prediction to default, though other factors reduce risk. For fair lending, consider whether credit history length penalizes certain demographic groups; consider supplementing the model with alternative data (rent payments, utilities) to improve credit signals for thin-file applicants.

### Example Local SHAP Interpretation — High-Risk Case 3

Test index: 4230
Predicted probability of default: 0.81 (81%)
True label: 1

**Top positive contributors:**

- recent_delinquencies (SHAP +0.60): Several recent late payments — strong signal toward default.

- interest_rate (SHAP +0.25): Elevated interest compared to peer group.

- employment_gap_months (SHAP +0.10): Recent job gaps increased uncertainty.

**Top negative contributors:**

- savings_balance (SHAP -0.15): Reasonable savings reduce immediate liquidity risk.

- co_applicant_present (SHAP -0.06): Co-signer reduces default likelihood.

**Interpretation & action:**
Recent delinquencies dominate the prediction. Actions: require additional documentation, consider conditional approval with higher monitoring or require cosigner/collateral, or provide tailored repayment plans.

---








