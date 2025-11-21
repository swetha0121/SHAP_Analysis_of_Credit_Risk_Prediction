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
