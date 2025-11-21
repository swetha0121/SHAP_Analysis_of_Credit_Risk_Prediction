"""
credit_risk_shap_full.py
End-to-end credit risk prediction pipeline with XGBoost and SHAP.
Saves:
 - plots/ (shap summary, beeswarm, force plots)
 - reports/ (classification report, metrics summary, feature importances, shap values for chosen cases)
 - models/ (best_model.joblib)
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import joblib

# ---------------------------
# Configuration / File paths
# ---------------------------
DATA_PATH = "credit_risk_dataset.csv"   # <- Update this path if needed
OUT_DIR = "outputs"
PLOTS_DIR = os.path.join(OUT_DIR, "plots")
REPORTS_DIR = os.path.join(OUT_DIR, "reports")
MODELS_DIR = os.path.join(OUT_DIR, "models")

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

RANDOM_STATE = 42

# ---------------------------
# Step 1: Load Dataset
# ---------------------------
print("Loading data from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print("Columns:\n", df.columns.tolist())
print(df.head().T)

# ---------------------------
# Step 2: Preprocessing
# ---------------------------
# 1) Missing values check and simple imputation
print("\nMissing values per column:")
print(df.isnull().sum())

# Backfill then forward fill as fallback (safe for many tabular datasets)
df.fillna(method='bfill', inplace=True)
df.fillna(method='ffill', inplace=True)

# 2) Identify categorical columns and encode
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
print("\nCategorical columns detected:", categorical_cols)

le_map = {}
for col in categorical_cols:
    le = LabelEncoder()
    try:
        df[col] = le.fit_transform(df[col].astype(str))
        le_map[col] = le
    except Exception as e:
        print(f"Warning encoding {col}: {e}")

# 3) Check target column exists
TARGET = "loan_status"
if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found in dataset. Please update TARGET variable.")

X = df.drop(columns=[TARGET])
y = df[TARGET]

# ---------------------------
# Step 3: Train-test split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print("\nTrain/Test sizes:", X_train.shape, X_test.shape)

# ---------------------------
# Step 4: Handle imbalance using SMOTE
# ---------------------------
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE class counts:")
print(pd.Series(y_train_res).value_counts())

# ---------------------------
# Step 5: Optional scaling (not necessary for tree models)
# ---------------------------
scaler = StandardScaler()
# We'll scale only if user wants; but keep original feature names for SHAP by using DataFrame later
X_train_res_scaled = scaler.fit_transform(X_train_res)
X_test_scaled = scaler.transform(X_test)

# We will use the unscaled features for XGBoost (trees don't require scaling). Keep scaled copies if needed.
X_train_used = X_train_res
X_test_used = X_test

# ---------------------------
# Step 6: Train XGBoost with GridSearchCV
# ---------------------------
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='auc',
    random_state=RANDOM_STATE,
    n_jobs=-1
)

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1]
}

grid = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=3,
    verbose=1,
    n_jobs=-1
)

print("\nStarting GridSearchCV...")
grid.fit(X_train_used, y_train_res)
best_model = grid.best_estimator_
print("GridSearchCV done. Best params:", grid.best_params_)

# Save model
joblib.dump(best_model, os.path.join(MODELS_DIR, "best_xgb_model.joblib"))
print("Saved best model to models/best_xgb_model.joblib")

# ---------------------------
# Step 7: Evaluation on test set
# ---------------------------
y_pred = best_model.predict(X_test_used)
y_proba = best_model.predict_proba(X_test_used)[:, 1]

auc = roc_auc_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred, zero_division=0)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
clf_report = classification_report(y_test, y_pred, digits=4)

print("\n=== Model Performance on Test Set ===")
print(f"AUC: {auc:.4f}")
print(f"F1: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nClassification Report:\n", clf_report)

# Save basic metrics to a file
with open(os.path.join(REPORTS_DIR, "metrics_summary.txt"), "w") as f:
    f.write("Model performance on test set\n")
    f.write(f"AUC: {auc:.6f}\n")
    f.write(f"F1: {f1:.6f}\n")
    f.write(f"Precision: {precision:.6f}\n")
    f.write(f"Recall: {recall:.6f}\n\n")
    f.write("Classification Report:\n")
    f.write(clf_report)

# Also save confusion matrix
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, index=["Actual_0","Actual_1"], columns=["Pred_0","Pred_1"])
cm_df.to_csv(os.path.join(REPORTS_DIR, "confusion_matrix.csv"))
print("Saved confusion matrix and metrics to reports/")

# Save the classification report text file
with open(os.path.join(REPORTS_DIR, "classification_report.txt"), "w") as f:
    f.write(clf_report)

# ---------------------------
# Step 8: Feature importances (XGBoost native)
# ---------------------------
feature_importances = pd.Series(best_model.feature_importances_, index=X_train_used.columns).sort_values(ascending=False)
feature_importances.to_csv(os.path.join(REPORTS_DIR, "xgb_feature_importances.csv"), header=["importance"])
print("Saved XGBoost native feature importances to reports/xgb_feature_importances.csv")

# Save a simple bar plot of feature importances (top 20)
plt.figure(figsize=(8,6))
feature_importances.head(20).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("XGBoost Feature Importances (top 20)")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, "xgb_feature_importance_top20.png"), dpi=300)
plt.close()

# ---------------------------
# Step 9: SHAP analysis
# ---------------------------
print("\nStarting SHAP analysis (this may take time depending on test set size)...")

# Use TreeExplainer for tree models
explainer = shap.TreeExplainer(best_model)
# shap_values shape: (n_samples, n_features)
shap_values = explainer.shap_values(X_test_used)

# Save shap values for test set (as CSV can be large) - we'll save only for top N instances and full small sampled subset
shap_df = pd.DataFrame(shap_values, columns=X_test_used.columns)
shap_df.to_csv(os.path.join(REPORTS_DIR, "shap_values_testset.csv"), index=False)
print("Saved SHAP values (test set partial/full) to reports/shap_values_testset.csv")

# Global SHAP plots
# 1) Bar (mean absolute shap)
plt.figure()
shap.summary_plot(shap_values, X_test_used, plot_type="bar", show=False)
plt.savefig(os.path.join(PLOTS_DIR, "shap_summary_bar.png"), dpi=300, bbox_inches='tight')
plt.close()

# 2) Beeswarm
plt.figure()
shap.summary_plot(shap_values, X_test_used, show=False)
plt.savefig(os.path.join(PLOTS_DIR, "shap_beeswarm.png"), dpi=300, bbox_inches='tight')
plt.close()

print("Saved SHAP global plots to plots/")

# ---------------------------
# Step 10: Local explanations — select 3 high-risk cases
# ---------------------------
# Choose top 3 instances by predicted probability of default (descending)
high_risk_idx = np.argsort(y_proba)[-3:][::-1]  # descending order
high_risk_idx_list = list(high_risk_idx)
print("High risk selected indices (test set positional indices):", high_risk_idx_list)

# Save SHAP contributions for selected high-risk cases (as CSV)
high_risk_shap = shap_df.iloc[high_risk_idx_list].copy()
high_risk_shap["pred_proba"] = y_proba[high_risk_idx_list]
high_risk_shap["true_label"] = y_test.reset_index(drop=True).iloc[high_risk_idx_list].values
high_risk_shap.to_csv(os.path.join(REPORTS_DIR, "high_risk_shap_contributions.csv"), index=True)

# Save force plots as PNG using matplotlib mode
for i, idx in enumerate(high_risk_idx_list, start=1):
    plt.figure()
    # Matplotlib force_plot (slow for many features): use matplotlib=True
    shap.force_plot(explainer.expected_value, shap_values[idx], X_test_used.iloc[idx], matplotlib=True, show=False)
    plt.title(f"SHAP force plot - high risk case #{i} (test idx {idx})")
    plt.tight_layout()
    savefp = os.path.join(PLOTS_DIR, f"shap_force_plot_highrisk_{i}_idx_{idx}.png")
    plt.savefig(savefp, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved:", savefp)

# Also produce a textual summary for each selected instance (top contributing features)
text_summaries = []
for i, idx in enumerate(high_risk_idx_list, start=1):
    row_shap = shap_df.iloc[idx].sort_values(ascending=False)
    top_pos = row_shap[row_shap > 0].head(5)
    top_neg = row_shap[row_shap < 0].abs().head(5)  # magnitude of negative contributors
    summary = f"Instance {i} (test idx {idx}, pred_proba={y_proba[idx]:.4f}, true_label={int(y_test.reset_index(drop=True).iloc[idx])}):\n"
    summary += "Top positive contributors (pushing towards default):\n"
    for feat, val in top_pos.items():
        summary += f"  - {feat}: SHAP={val:.4f}\n"
    summary += "Top negative contributors (pushing against default):\n"
    for feat, val in top_neg.items():
        summary += f"  - {feat}: SHAP=-{val:.4f}\n"
    text_summaries.append(summary)

# Save the textual local summaries
with open(os.path.join(REPORTS_DIR, "local_shap_summaries.txt"), "w") as f:
    f.write("\n\n".join(text_summaries))

print("Saved textual local SHAP summaries to reports/local_shap_summaries.txt")

# ---------------------------
# Step 11: Auto-generate a human-readable report file (Deliverable #2)
# ---------------------------
report_path = os.path.join(REPORTS_DIR, "auto_report.txt")
with open(report_path, "w") as f:
    f.write("Interpretable Machine Learning: SHAP Analysis of Credit Risk Prediction\n")
    f.write("="*80 + "\n\n")
    f.write("1) Model & Training Summary\n")
    f.write("---------------------------\n")
    f.write(f"Model: XGBoost (xgb.XGBClassifier)\n")
    f.write(f"Best hyperparameters (GridSearchCV): {grid.best_params_}\n\n")
    f.write("2) Performance on Test Set\n")
    f.write("---------------------------\n")
    f.write(f"AUC: {auc:.6f}\n")
    f.write(f"F1: {f1:.6f}\n")
    f.write(f"Precision: {precision:.6f}\n")
    f.write(f"Recall: {recall:.6f}\n\n")
    f.write("Classification Report:\n")
    f.write(clf_report + "\n")
    f.write("Confusion matrix saved at: reports/confusion_matrix.csv\n\n")
    f.write("3) Global Feature Importance\n")
    f.write("---------------------------\n")
    f.write("XGBoost native importances saved at: reports/xgb_feature_importances.csv\n")
    f.write("SHAP global plots saved at: outputs/plots/shap_summary_bar.png and outputs/plots/shap_beeswarm.png\n\n")
    f.write("4) Local Explanations (Top 3 high-risk cases)\n")
    f.write("---------------------------\n")
    f.write("Local SHAP contribution CSV: reports/high_risk_shap_contributions.csv\n")
    f.write("Local textual summaries: reports/local_shap_summaries.txt\n")
    f.write("\n\n")
    f.write("NOTE: For regulatory write-ups, include the SHAP plots and local summaries for each contested application.\n")

print("Auto-report written to:", report_path)

print("\nAll done. Check the 'outputs' folder for saved models, plots, and reports.")
