# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, classification_report, confusion_matrix

from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE

import xgboost as xgb
import shap

# Step 2: Load Dataset
df = pd.read_csv("credit_risk_dataset.csv")  # Replace with your downloaded CSV path
print("Dataset shape:", df.shape)
print(df.head())

# Step 3: Data Preprocessing
# Check missing values
print(df.isnull().sum())

# Fill missing values or drop rows/columns if necessary
df.fillna(method='bfill', inplace=True)

print(df.columns)
# Encode categorical variables
categorical_cols = df.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Step 4: Split Features and Target
X = df.drop("loan_status", axis=1)  # Replace 'loan_status' with target column name
y = df["loan_status"]

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("After SMOTE, counts of label '1':", sum(y_train_res==1))
print("After SMOTE, counts of label '0':", sum(y_train_res==0))


# Step 7: Feature Scaling (optional for tree-based models, mostly for interpretability)
scaler = StandardScaler()
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# Step 8: Train XGBoost Classifier with hyperparameter tuning
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42
)

params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1]
}

grid = GridSearchCV(xgb_model, param_grid=params, cv=3, scoring='roc_auc', verbose=1)
grid.fit(X_train_res, y_train_res)

best_model = grid.best_estimator_
print("Best Parameters:", grid.best_params_)

# Step 9: Model Evaluation
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Step 10: SHAP Analysis
explainer = shap.TreeExplainer(best_model.get_booster())
shap_values = explainer.shap_values(X_test)

# Global feature importance
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)  # Detailed beeswarm plot

# Step 11: Local explanations for 3 high-risk cases
# Identify top 3 predicted defaults with highest probability
high_risk_idx = np.argsort(y_proba)[-3:]
for idx in high_risk_idx:
    print(f"\nLocal explanation for test instance {idx}:")
    shap.force_plot(explainer.expected_value, shap_values[idx], X_test.iloc[idx], matplotlib=True)

# Step 12: Textual Analysis
# Example of extracting SHAP values for textual explanation
feature_contributions = pd.DataFrame(shap_values, columns=X_test.columns)
instance_contribs = feature_contributions.iloc[high_risk_idx]
print("\nFeature contributions for high-risk cases:")
print(instance_contribs)

# XGBoost built-in feature importance
importances = best_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X_test.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# SHAP global importance (already done via summary_plot)
shap.summary_plot(shap_values, X_test, plot_type="bar")

