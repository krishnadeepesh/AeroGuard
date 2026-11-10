"""
AeroGuard - Model Training Pipeline
Trains:
1. Baseline: Multinomial Logistic Regression
2. Primary Model: Random Forest Classifier
Evaluates on unseen test set and serializes trained models to disk.
"""

import os
import json
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_score

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.data_loader import prepare_train_test_data, FEATURE_COLUMNS, AQI_CLASSES

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")

def train_and_evaluate_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_train_test_data()

    print("\n--- 1. Training Baseline: Multinomial Logistic Regression ---")
    baseline_clf = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=42)
    baseline_clf.fit(X_train_scaled, y_train)

    baseline_preds = baseline_clf.predict(X_test_scaled)
    baseline_acc = accuracy_score(y_test, baseline_preds)
    baseline_f1_macro = f1_score(y_test, baseline_preds, average="macro")
    baseline_f1_weighted = f1_score(y_test, baseline_preds, average="weighted")
    print(f"Baseline Accuracy: {baseline_acc:.4f} | Macro F1: {baseline_f1_macro:.4f}")

    print("\n--- 2. Training Primary Model: Random Forest Classifier ---")
    rf_clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=16,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf_clf.fit(X_train_scaled, y_train)

    rf_preds = rf_clf.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_precision_macro = precision_score(y_test, rf_preds, average="macro", zero_division=0)
    rf_recall_macro = recall_score(y_test, rf_preds, average="macro", zero_division=0)
    rf_f1_macro = f1_score(y_test, rf_preds, average="macro", zero_division=0)
    rf_f1_weighted = f1_score(y_test, rf_preds, average="weighted", zero_division=0)

    # 5-Fold Cross Validation on Training Set to demonstrate generalizability
    cv_scores = cross_val_score(rf_clf, X_train_scaled, y_train, cv=5, scoring="f1_macro", n_jobs=-1)
    print(f"Random Forest Test Accuracy : {rf_acc:.4f}")
    print(f"Random Forest Macro Precision: {rf_precision_macro:.4f}")
    print(f"Random Forest Macro Recall   : {rf_recall_macro:.4f}")
    print(f"Random Forest Macro F1       : {rf_f1_macro:.4f}")
    print(f"5-Fold CV Mean Macro F1     : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Feature Importances
    feature_importances = dict(zip(FEATURE_COLUMNS, [round(float(v), 4) for v in rf_clf.feature_importances_]))
    print("\nFeature Importances:", feature_importances)

    # Confusion Matrix
    cm = confusion_matrix(y_test, rf_preds, labels=AQI_CLASSES)
    cm_baseline = confusion_matrix(y_test, baseline_preds, labels=AQI_CLASSES)

    # Classification Reports
    report_rf = classification_report(y_test, rf_preds, labels=AQI_CLASSES, output_dict=True, zero_division=0)
    report_baseline = classification_report(y_test, baseline_preds, labels=AQI_CLASSES, output_dict=True, zero_division=0)

    # Serialize Models
    rf_path = os.path.join(MODELS_DIR, "aqi_random_forest.joblib")
    baseline_path = os.path.join(MODELS_DIR, "aqi_logistic_regression.joblib")
    joblib.dump(rf_clf, rf_path)
    joblib.dump(baseline_clf, baseline_path)

    # Save comprehensive evaluation results
    eval_results = {
        "classes": AQI_CLASSES,
        "features": FEATURE_COLUMNS,
        "baseline_logistic_regression": {
            "accuracy": round(float(baseline_acc), 4),
            "macro_f1": round(float(baseline_f1_macro), 4),
            "weighted_f1": round(float(baseline_f1_weighted), 4),
            "confusion_matrix": cm_baseline.tolist(),
            "report": report_baseline
        },
        "primary_random_forest": {
            "accuracy": round(float(rf_acc), 4),
            "macro_precision": round(float(rf_precision_macro), 4),
            "macro_recall": round(float(rf_recall_macro), 4),
            "macro_f1": round(float(rf_f1_macro), 4),
            "weighted_f1": round(float(rf_f1_weighted), 4),
            "cv_5fold_mean_f1": round(float(cv_scores.mean()), 4),
            "cv_5fold_std": round(float(cv_scores.std()), 4),
            "feature_importances": feature_importances,
            "confusion_matrix": cm.tolist(),
            "report": report_rf
        }
    }

    eval_path = os.path.join(MODELS_DIR, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n[Models] Saved trained models and evaluation metrics to: {MODELS_DIR}")
    return eval_results

if __name__ == "__main__":
    train_and_evaluate_models()
