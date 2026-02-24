#!/usr/bin/env python
# coding: utf-8

# # Fit & Evaluate RF for NOVA group
# Fit on full training set and evaluate on held out test set

# In[1]:


import os
import time
import numpy as np
import pandas as pd
import yaml

from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


# In[3]:


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

# 3. Target – NOVA group
target = "nova_group"

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

# 5. Load parameters from params.yaml
with open("params.yaml", "r") as f:
    config = yaml.safe_load(f)

# Shared parameters for all RF runs
base_rf_params = config["rf_nova"]["base"]

# Phase-specific overrides (matching the phase names in your 'phases' dict)
best_params_by_phase = {
    "phase1_intrinsic": config["rf_nova"]["phase1"],
    "phase2_intr+extr": config["rf_nova"]["phase2"],
    "phase3_all_labels": config["rf_nova"]["phase3"]
}

# 6. Containers for results
all_metrics = []
coverage_stats = [] 

# 7. Loop over phases – fit/evaluate only
for phase_name, label_col in phases.items():
    print("\n" + "="*90)
    print(f"=== {phase_name} – Using column '{label_col}' ===")
    print("="*90)

    # 7.1 Encode label features for this phase
    train[label_col] = train[label_col].fillna("")
    test[label_col]  = test[label_col].fillna("")

    train_labels = train[label_col].apply(lambda x: x.split("|") if x != "" else [])
    test_labels  = test[label_col].apply(lambda x: x.split("|") if x != "" else [])

    mlb = MultiLabelBinarizer(sparse_output=True)
    X_train_labels = mlb.fit_transform(train_labels)
    X_test_labels  = mlb.transform(test_labels)

    print(f"Encoded label matrix shape (train): {X_train_labels.shape}")
    print(f"Encoded label matrix shape (test):  {X_test_labels.shape}")

    #  PhaseCoverage (at least 1 label in this phase)
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

    # 7.2 Fit Random Forest with best params for this phase
    rf_params = {**base_rf_params, **best_params_by_phase[phase_name]}
    rf = RandomForestClassifier(**rf_params)

    fit_start = time.perf_counter()
    rf.fit(X_train_labels, y_train)  # train on ALL rows
    fit_end = time.perf_counter()
    fit_time = fit_end - fit_start

    print(f"Fitted Random Forest for {phase_name} in {fit_time:.2f} s")

    # 7.3 Predictions: Train–Overall, Test–Overall, Test–Core
    # Train–Overall
    y_pred_train = rf.predict(X_train_labels)

    # Test–Overall
    y_pred_test = rf.predict(X_test_labels)

    # Test–Core subset
    y_test_core = y_test[core_mask]
    y_pred_test_core = y_pred_test[core_mask]

    # PhaseCoverage subsets (at least 1 label in this phase)
    y_train_cov = y_train[train_nonempty_mask]
    y_pred_train_cov = y_pred_train[train_nonempty_mask]

    y_test_cov = y_test[test_nonempty_mask]
    y_pred_test_cov = y_pred_test[test_nonempty_mask]

    # 7.4 Attach predictions back to DataFrames (model-specific cols)
    train[f"{phase_name}_rf_pred_enc"] = y_pred_train
    test[f"{phase_name}_rf_pred_enc"] = y_pred_test

    train[f"{phase_name}_rf_pred"] = le.inverse_transform(y_pred_train)
    test[f"{phase_name}_rf_pred"] = le.inverse_transform(y_pred_test)

    # 7.5 Compute metrics helper
    def compute_metrics(y_true, y_pred, phase, dataset_name):
        if len(y_true) == 0:
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

        return {
            "Phase": phase,
            "Dataset": dataset_name,
            "n_samples": len(y_true),
            "Accuracy": accuracy_score(y_true, y_pred),
            "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Precision (Macro)": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "Recall (Macro)": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "F1 (Macro)": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "F1 (Micro)": f1_score(y_true, y_pred, average="micro", zero_division=0),
        }

    # Train metrics
    all_metrics.append(compute_metrics(y_train, y_pred_train, phase_name, "Train-Overall"))
    all_metrics.append(compute_metrics(y_train_cov, y_pred_train_cov, phase_name, "Train-PhaseCoverage"))
    # Test metrics
    all_metrics.append(compute_metrics(y_test, y_pred_test, phase_name, "Test-Overall"))
    all_metrics.append(compute_metrics(y_test_cov,  y_pred_test_cov,  phase_name, "Test-PhaseCoverage"))
    all_metrics.append(compute_metrics(y_test_core, y_pred_test_core, phase_name, "Test-Core"))

    # Print quick summary for this phase (last 5 entries)
    print("\nMetrics –", phase_name)
    for m in all_metrics[-5:]:
        print(
            f"{m['Dataset']}: n={m['n_samples']}, "
            f"Acc={m['Accuracy']:.4f}, "
            f"BalAcc={m['Balanced Accuracy']:.4f}, "
            f"F1_macro={m['F1 (Macro)']:.4f}"
        )

# 8. Combine and inspect metrics across phases & datasets
metrics_df = pd.DataFrame(all_metrics)
coverage_df = pd.DataFrame(coverage_stats)

print("\n=== All Metrics (Train-Overall, Test-Overall, Test-Core, PhaseCoverage) ===")
print(metrics_df)

print("\n=== Coverage stats per phase & dataset ===")
print(coverage_df)

# 9. Save outputs
metrics_df.to_csv("results/rf_nova_eval_metrics_train_test_core.csv", index=False)
coverage_df.to_csv("results/rf_nova_phase_coverage_stats.csv", sep=";", index=False)
train.to_csv("results/data_def_train_with_rf_nova_preds.csv", sep=";", index=False)
test.to_csv("results/data_def_test_with_rf_nova_preds.csv", sep=";", index=False)

print("\nFiles written to ./results/:")
print(" - rf_nova_eval_metrics_train_test_core.csv")
print(" - rf_nova_phase_coverage_stats.csv")
print(" - data_def_train_with_rf_nova_preds.csv")
print(" - data_def_test_with_rf_nova_preds.csv")

