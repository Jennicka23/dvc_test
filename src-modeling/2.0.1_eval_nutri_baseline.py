#!/usr/bin/env python
# coding: utf-8

# # Fit & Evaluate Baseline for Nutri-Score
# Fit on full training set and evaluate on held out test set

# In[ ]:


import pandas as pd
import numpy as np
import time
import yaml

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# In[7]:


# 1. Load data
train = pd.read_csv("gen/data_def_train_folds.csv")
test  = pd.read_csv("gen/data_def_test.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)

# 2. Create combined column for Phase 2 (intrinsic + extrinsic)
train["labels_1_2_intr_extr"] = (
    train[["labels_1_intrinsic", "labels_2_extrinsic"]]
    .fillna("")
    .agg("|".join, axis=1)
    .str.strip("|")
)

test["labels_1_2_intr_extr"] = (
    test[["labels_1_intrinsic", "labels_2_extrinsic"]]
    .fillna("")
    .agg("|".join, axis=1)
    .str.strip("|")
)

# 3. Target (change only this for NOVA later)
target = "nutriscore"  

# Encode target variable
le = LabelEncoder()
y_train = le.fit_transform(train[target])
y_test = le.transform(test[target])

print("Class mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

# 4. Define phases (name -> column)
phases = {
    "phase1_intrinsic": "labels_1_intrinsic",
    "phase2_intr+extr": "labels_1_2_intr_extr",
    "phase3_all_labels": "labels_string",
}

# Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)
    
logreg_params = params["baseline"]

# 5. Containers for results
all_metrics = []  
coverage_stats = []  

# 6. Loop over phases – fit/evaluate only (no CV)
for phase_name, label_col in phases.items():
    print("\n" + "="*80)
    print(f"=== {phase_name} – using column '{label_col}' ===")
    print("="*80)

    # 6.1 Prepare label features for this phase
    train[label_col] = train[label_col].fillna("")
    test[label_col]  = test[label_col].fillna("")

    train_labels = train[label_col].apply(lambda x: x.split("|") if x != "" else [])
    test_labels  = test[label_col].apply(lambda x: x.split("|") if x != "" else [])

    mlb = MultiLabelBinarizer(sparse_output=True)
    X_train_labels = mlb.fit_transform(train_labels)
    X_test_labels  = mlb.transform(test_labels)

    print(f"Encoded label matrix shape (train): {X_train_labels.shape}")
    print(f"Encoded label matrix shape (test):  {X_test_labels.shape}")

    # PhaseCoverage (at least 1 label in this phase)
    train_nonempty_mask = np.array(X_train_labels.sum(axis=1)).ravel() > 0
    test_nonempty_mask  = np.array(X_test_labels.sum(axis=1)).ravel() > 0

    # Basic counts for coverage
    train_total = len(y_train)
    test_total  = len(y_test)
    core_mask   = test["core_slice"].astype(bool).values
    core_total  = int(core_mask.sum())

    train_cov_n = int(train_nonempty_mask.sum())
    test_cov_n  = int(test_nonempty_mask.sum())
    core_cov_n  = int((test_nonempty_mask & core_mask).sum())

    # Store coverage stats
    coverage_stats.append({
        "Phase": phase_name,
        "Dataset": "Train",
        "n_total": train_total,
        "n_with_labels": train_cov_n,
        "frac_with_labels": train_cov_n / train_total if train_total > 0 else np.nan,
    })
    coverage_stats.append({
        "Phase": phase_name,
        "Dataset": "Test",
        "n_total": test_total,
        "n_with_labels": test_cov_n,
        "frac_with_labels": test_cov_n / test_total if test_total > 0 else np.nan,
    })
    coverage_stats.append({
        "Phase": phase_name,
        "Dataset": "Test-Core",
        "n_total": core_total,
        "n_with_labels": core_cov_n,
        "frac_with_labels": core_cov_n / core_total if core_total > 0 else np.nan,
    })

    print(
        f"Coverage – {phase_name}:\n"
        f"  Train:     {train_cov_n}/{train_total} "
        f"({train_cov_n/train_total:.3%} with ≥1 label in this phase)\n"
        f"  Test:      {test_cov_n}/{test_total} "
        f"({test_cov_n/test_total:.3%} with ≥1 label in this phase)\n"
        f"  Test-Core: {core_cov_n}/{core_total} "
        f"({core_cov_n/core_total:.3%} with ≥1 label in this phase)"
    )

    # 6.2 Fit on full training set
    logreg = LogisticRegression(**logreg_params)

    fit_start = time.perf_counter()
    logreg.fit(X_train_labels, y_train)
    fit_end = time.perf_counter()
    fit_time = fit_end - fit_start

    print(f"Fitted LogisticRegression for {phase_name} in {fit_time:.2f} s")

    # 6.3 Predictions: Train–Overall, Test–Overall, Test–Core
    # Train–Overall
    pred_train_start = time.perf_counter()
    y_pred_train = logreg.predict(X_train_labels)
    pred_train_end = time.perf_counter()
    pred_train_time = pred_train_end - pred_train_start

    # Test–Overall
    pred_test_start = time.perf_counter()
    y_pred_test = logreg.predict(X_test_labels)
    pred_test_end = time.perf_counter()
    pred_test_time = pred_test_end - pred_test_start

    # Test–Core: subset where core_slice == True
    if core_mask.sum() == 0:
        print("Warning: no rows with core_slice == True in test set.")
        y_test_core = np.array([])
        y_pred_test_core = np.array([])
    else:
        y_test_core = y_test[core_mask]
        y_pred_test_core = y_pred_test[core_mask]

    # PhaseCoverage subsets (at least 1 label in this phase)
    y_train_cov = y_train[train_nonempty_mask]
    y_pred_train_cov = y_pred_train[train_nonempty_mask]

    y_test_cov = y_test[test_nonempty_mask]
    y_pred_test_cov = y_pred_test[test_nonempty_mask]

    # 6.4 Attach predictions back to train/test DataFrames
    # numeric predictions (encoded)
    train[f"{phase_name}_pred_enc"] = y_pred_train
    test[f"{phase_name}_pred_enc"] = y_pred_test

    # decoded predictions (original class labels, e.g. 'A','B','C','D','E')
    train[f"{phase_name}_pred"] = le.inverse_transform(y_pred_train)
    test[f"{phase_name}_pred"] = le.inverse_transform(y_pred_test)

    # 6.5 Compute metrics for each dataset view
    def compute_metrics(y_true, y_pred, phase, dataset_name):
        """Return a dict of metrics for convenience."""
        if len(y_true) == 0:
            # In case a slice ends up empty, return NaNs
            return {
                "Phase": phase,
                "Dataset": dataset_name,
                "n_samples": 0,
                "Accuracy": np.nan,
                "Balanced Accuracy": np.nan,
                "Precision (Macro)": np.nan,
                "Recall (Macro)": np.nan,
                "F1 (Macro)": np.nan,
                "F1 (Micro)": np.nan,
            }

        acc = accuracy_score(y_true, y_pred)
        bal_acc = balanced_accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_mic = f1_score(y_true, y_pred, average="micro", zero_division=0)

        return {
            "Phase": phase,
            "Dataset": dataset_name,
            "n_samples": len(y_true),
            "Accuracy": acc,
            "Balanced Accuracy": bal_acc,
            "Precision (Macro)": prec,
            "Recall (Macro)": rec,
            "F1 (Macro)": f1_mac,
            "F1 (Micro)": f1_mic,
        }

    # Train Metrics
    metrics_train = compute_metrics(y_train, y_pred_train, phase_name, "Train-Overall")
    all_metrics.append(metrics_train)

    metrics_train_cov = compute_metrics(y_train_cov, y_pred_train_cov, phase_name, "Train-PhaseCoverage")
    all_metrics.append(metrics_train_cov)

    # Test Metrics
    metrics_test = compute_metrics(y_test, y_pred_test, phase_name, "Test-Overall")
    all_metrics.append(metrics_test)

    metrics_test_cov  = compute_metrics(y_test_cov,  y_pred_test_cov,  phase_name, "Test-PhaseCoverage")
    all_metrics.append(metrics_test_cov)

    metrics_core = compute_metrics(y_test_core, y_pred_test_core, phase_name, "Test-Core")
    all_metrics.append(metrics_core)

    # Print summary for quick inspection
    print("\nMetrics –", phase_name)
    for m in [metrics_train, metrics_test, metrics_core, metrics_train_cov, metrics_test_cov]:
        print(
            f"{m['Dataset']}: "
            f"n={m['n_samples']}, "
            f"Acc={m['Accuracy']:.4f} "
            f"(Bal={m['Balanced Accuracy']:.4f}, "
            f"F1_macro={m['F1 (Macro)']:.4f})"
        )

# 7. Combine and inspect metrics across phases & datasets
metrics_df = pd.DataFrame(all_metrics)
coverage_df = pd.DataFrame(coverage_stats)

print("\n=== All Metrics (Train-Overall, Test-Overall, Test-Core, PhaseCoverage) ===")
print(metrics_df)

print("\n=== Coverage stats per phase & dataset ===")
print(coverage_df)

# 8. Save outputs
metrics_df.to_csv("results/logreg_nutri_eval_metrics_train_test_core.csv", sep=";", index=False)
coverage_df.to_csv("results/logreg_nutri_phase_coverage_stats.csv", sep=";", index=False)
train.to_csv("results/data_def_train_with_baseline_nutri_preds.csv", sep=";", index=False)
test.to_csv("results/data_def_test_with_baseline_nutri_preds.csv", sep=";", index=False)

print("\nFiles written:")
print(" - logreg_nutri_eval_metrics_train_test_core.csv")
print(" - logreg_nutri_phase_coverage_stats.csv")
print(" - data_def_train_with_baseline_nutri_preds.csv")
print(" - data_def_test_with_baseline_nutri_preds.csv")

